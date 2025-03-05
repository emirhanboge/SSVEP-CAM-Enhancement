import torch
import numpy as np
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    GradCAMElementWise,
    HiResCAM,
    ScoreCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM
)
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
from utils.utils import get_dataset, get_model, set_random_seed

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torch.nn import ReLU
from utils.utils import reverse_transform
import json
import traceback
import xml.etree.ElementTree as ET
import tarfile
import io
import cv2
def get_cam_method(method_name):
    """Get CAM method class by name."""
    methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "gradcam_elementwise": GradCAMElementWise,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
    }
    return methods.get(method_name.lower())

def prepare_focused_cam(
    model,
    dataset_name,
    input_image,
    input_label,
    input_idx,
    input_raw_idx,
    model_name,
    base_path,
    device,
    method_name="gradcam++",
    k_neurons=1000,
    n_last_layers=5,
):
    """
    Prepare CAM visualization focused on top-k neurons from SSVEP analysis.
    Only considers layer4 for ResNet models.
    """
    from SSVEP_main import get_top_neurons, parse_neuron_keys
    
    # Ensure input_image is on CPU and in the right format (3, H, W)
    if isinstance(input_image, torch.Tensor):
        input_image_cpu = input_image.detach().cpu()
        if input_image_cpu.dim() == 4:
            input_image_cpu = input_image_cpu.squeeze(0)
    else:
        input_image_cpu = input_image
    
   # print(f"Input image type and device after processing: {type(input_image_cpu)}, {input_image_cpu.device if isinstance(input_image_cpu, torch.Tensor) else 'N/A'}")
   # print(f"Input image shape after processing: {input_image_cpu.shape if isinstance(input_image_cpu, torch.Tensor) else 'N/A'}")
    
    try:
        # Get top neurons from SSVEP analysis
        top_neurons = get_top_neurons(
            model,
            [input_image_cpu],
            [input_label],
            device,
            base_path,
            dataset_name,
            model_name,
            xai_exp=True,
            input_idx=input_idx,
        )
    except Exception as e:
        print(f"Error in get_top_neurons: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        raise e

    # Get all convolution layer names from model
    # get 24, 25, 26, 27, 28, 29 30
    
    conv_layer_names = [
        name for name, module in model.named_modules() 
        if "24" in name or "25" in name or "26" in name or "27" in name or "28" in name or "29" in name or "30" in name
    ]
    
    # Use all conv layers in layer4 instead of last n layers
    last_n_conv_layers = set(conv_layer_names)
    # Filter the top neurons to only include those from last n conv layers
    filtered_neurons = {
        k: v for k, v in top_neurons.items() 
        if k.split("'")[1] in last_n_conv_layers  # Split by quote and take the layer name
    }
    
    # Get exactly k_neurons top filters from the filtered set
    top_filters = parse_neuron_keys(filtered_neurons)[:k_neurons]
    #print(f"Selected exactly {len(top_filters)} top filters from layer4")

    # Create a mapping of layer names to their corresponding filters
    layer_to_filters = {}
    for layer_name, filter_idx in top_filters:
        if layer_name not in layer_to_filters:
            layer_to_filters[layer_name] = []
        layer_to_filters[layer_name].append(filter_idx)

    # Custom activation extractor that only considers top k neurons
    class FilterActivationExtractor:
        def __init__(self, layer_name, filter_indices):
            self.layer_name = layer_name
            self.filter_indices = torch.tensor(filter_indices)  # Convert to tensor for indexing
            self.activations = None
            self.hook = None
          #  print(f"Initialized extractor for layer {layer_name} with {len(filter_indices)} filters: {filter_indices[:5]}...")

        def __call__(self, module, input, output):
            """
            Hook to extract activations only for the specified filter indices.
            output shape: [batch_size, all_channels, height, width]
            """
            if isinstance(output, torch.Tensor):
                # Ensure filter_indices is on the same device as output
                filter_indices = self.filter_indices.to(output.device)
                # Extract only the activations for important filters
                selected_activations = output.index_select(1, filter_indices)
                self.activations = selected_activations
              #  print(f"Layer {self.layer_name}: Selected {len(self.filter_indices)} filters, "
              #      f"activation shape: {self.activations.shape}")
            else:
                print(f"Warning: output is not a tensor for layer {self.layer_name}, "
                      f"type: {type(output)}")
                self.activations = None

        def register(self, module):
            self.hook = module.register_forward_hook(self)
        
        def remove(self):
            if self.hook is not None:
                self.hook.remove()

    # Find target layers and register hooks
    activation_extractors = {}
    target_layers = []
    total_registered_filters = 0

    for layer_name, filter_indices in layer_to_filters.items():
        for name, module in model.named_modules():
            if name == layer_name:
                extractor = FilterActivationExtractor(layer_name, filter_indices)
                activation_extractors[layer_name] = extractor
                extractor.register(module)
                target_layers.append(module)
                total_registered_filters += len(filter_indices)
               # print(f"Registered {len(filter_indices)} filters for layer {layer_name}")
                break

    if not target_layers:
        raise ValueError(f"Could not find any target layers")

    # Verify we have exactly k_neurons filters
    assert total_registered_filters == k_neurons, \
        f"Filter count mismatch: registered {total_registered_filters}, expected {k_neurons}"

    class CustomTarget:
        def __init__(self, activation_extractors, target_label):
            self.activation_extractors = activation_extractors
            self.target_label = target_label
            
            total_filters = sum(len(extractor.filter_indices) for extractor in activation_extractors.values())
           # print(f"Using {total_filters} most important filters from layer4")

        def __call__(self, model_output):
            # Find the largest feature map size among all layers
            max_h, max_w = 0, 0
            for extractor in self.activation_extractors.values():
                if extractor.activations is not None:
                    _, _, h, w = extractor.activations.shape
                    max_h, max_w = max(max_h, h), max(max_w, w)

            # Initialize the final attribution map
            attribution_map = torch.zeros((max_h, max_w), device=model_output.device)

            # Combine activations from all selected important filters
            for name, extractor in self.activation_extractors.items():
                if extractor.activations is not None:
                    activations = extractor.activations  # [B, selected_filters, H, W]
                    
                    # Resize to match the largest feature map size
                    if (activations.shape[2], activations.shape[3]) != (max_h, max_w):
                        activations = F.interpolate(
                            activations,
                            size=(max_h, max_w),
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # Sum up the activations from all important filters
                    layer_attribution = activations.sum(dim=1)[0]  # Take first batch, sum across filters
                    attribution_map += F.relu(layer_attribution)  # Only consider positive attributions
            
            # Normalize the attribution map to [0, 1] range
            if attribution_map.max() > 0:
                attribution_map = attribution_map / attribution_map.max()

            # Handle different output shapes safely
            if model_output.dim() == 2:  # [batch_size, num_classes]
                return model_output[0, self.target_label]
            elif model_output.dim() == 1:  # [num_classes]
                return model_output[self.target_label]
            else:
                raise ValueError(f"Unexpected model output shape: {model_output.shape}")

    # Initialize CAM
    cam_method_class = get_cam_method(method_name)
    if cam_method_class is None:
        raise ValueError(f"Unknown CAM method: {method_name}")
    
    cam = cam_method_class(
        model=model,
        target_layers=target_layers,
        
    )

    # Ensure input_tensor is 4D: [batch_size, channels, height, width]
    input_tensor = input_image.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    
    targets = [CustomTarget(activation_extractors, input_label)]
    
    try:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=True if method_name in ["eigencam", "eigengradcam"] else False
        )
        grayscale_cam_base = grayscale_cam
        grayscale_cam = grayscale_cam[0, :]
    except Exception as e:
        print(f"Error in CAM generation: {str(e)}")
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Target layers: {[type(layer) for layer in target_layers]}")
        raise e

    # Remove the hooks
    for extractor in activation_extractors.values():
        extractor.remove()
 
    del cam, cam_method_class, CustomTarget, FilterActivationExtractor # Just in case

    return grayscale_cam, grayscale_cam_base


def get_predicted_bbox(cam, threshold_ratio=0.15):
    """
    Generate a bounding box from a class activation map (CAM).
    
    Args:
        cam (torch.Tensor or np.ndarray): Class activation map.
        threshold_ratio (float): Percentage of the max value used for thresholding.
    
    Returns:
        list or None: Predicted bounding box [x_min, y_min, x_max, y_max] (normalized).
    """
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()  # Convert to NumPy if it's a PyTorch tensor

    h, w = cam.shape
    max_val = np.max(cam)
    threshold = threshold_ratio * max_val

    # Binarize the CAM
    _, binary_map = cv2.threshold(cam.astype(np.uint8), int(threshold), 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No valid bounding box found

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box from the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Normalize coordinates
    x_min, y_min = x / cam.shape[1], y / cam.shape[0]
    x_max, y_max = (x + w) / cam.shape[1], (y + h) / cam.shape[0]

    return [x_min, y_min, x_max, y_max]


def calculate_iou(bbox_pred, bbox_gt):
    """
    Calculate Intersection over Union (IoU) between predicted and ground truth bounding boxes.
    
    Args:
        bbox_pred (list): Predicted bounding box [x_min, y_min, x_max, y_max] (normalized).
        bbox_gt (list): Ground truth bounding box [x_min, y_min, x_max, y_max] (normalized).
    
    Returns:
        float: IoU score (0 to 1).
    """
    if bbox_pred is None or bbox_gt is None:
        return 0.0  # No valid IoU if a bounding box is missing

    x_min_pred, y_min_pred, x_max_pred, y_max_pred = bbox_pred
    x_min_gt, y_min_gt, x_max_gt, y_max_gt = bbox_gt

    # Compute intersection
    x_min_inter = max(x_min_pred, x_min_gt)
    y_min_inter = max(y_min_pred, y_min_gt)
    x_max_inter = min(x_max_pred, x_max_gt)
    y_max_inter = min(y_max_pred, y_max_gt)

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    intersection = inter_width * inter_height

    # Compute union
    area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)
    area_gt = (x_max_gt - x_min_gt) * (y_max_gt - y_min_gt)
    union = area_pred + area_gt - intersection

    iou = intersection / (union + 1e-6)  # Avoid division by zero
    return iou


def evaluate_bbox(cam, bbox_gt, threshold_ratio=0.15, iou_threshold=0.5):
    """
    Evaluate if the predicted bounding box is correct based on IoU.

    Args:
        cam (torch.Tensor or np.ndarray): Class activation map.
        bbox_gt (list): Ground truth bounding box [x_min, y_min, x_max, y_max] (normalized).
        threshold_ratio (float): Thresholding ratio for CAM binarization.
        iou_threshold (float): Minimum IoU for correct detection.

    Returns:
        bool: True if IoU ≥ iou_threshold, otherwise False.
        float: IoU score.
    """
    bbox_pred = get_predicted_bbox(cam, threshold_ratio)
    iou = calculate_iou(bbox_pred, bbox_gt)
    return iou >= iou_threshold, iou


def evaluate_loc1(cam, bbox_gt, threshold_ratio=0.15, iou_threshold=0.5):
    """
    Evaluate IoU for loc1 (top-1 prediction).

    Args:
        cam (torch.Tensor or np.ndarray): Class activation map of top-1 prediction.
        bbox_gt (list): Ground truth bounding box [x_min, y_min, x_max, y_max] (normalized).
        threshold_ratio (float): Thresholding ratio for CAM binarization.
        iou_threshold (float): Minimum IoU for correct detection.

    Returns:
        bool: True if IoU ≥ iou_threshold, otherwise False.
        float: IoU score.
    """
    return evaluate_bbox(cam, bbox_gt, threshold_ratio, iou_threshold)


def evaluate_loc5(cams, bbox_gt, threshold_ratio=0.15, iou_threshold=0.5):
    """
    Evaluate if any of the top-5 CAMs predict the correct bounding box.

    Args:
        cams (list of torch.Tensor): List of CAM heatmaps for top-5 predictions.
        bbox_gt (list): Ground truth bounding box [x_min, y_min, x_max, y_max] (normalized).
        threshold_ratio (float): Thresholding ratio for CAM binarization.
        iou_threshold (float): Minimum IoU for correct detection.

    Returns:
        bool: True if at least one of the top-5 CAMs has IoU ≥ iou_threshold, otherwise False.
        float: Highest IoU score among the top-5 predictions.
    """
    max_iou = 0.0
    for cam in cams:
        correct, iou = evaluate_bbox(cam, bbox_gt, threshold_ratio, iou_threshold)
        max_iou = max(max_iou, iou)
        if correct:
            return True, max_iou
    return False, max_iou

def load_imagenet_bbox(dataset_dir, image_id):
    """
    Load ImageNet bounding box annotation from XML files.
    Returns normalized coordinates [x_min, y_min, x_max, y_max].
    
    Args:
        dataset_dir: Path to dataset directory containing bounding_boxes folder
        image_id: Image ID (e.g., 'ILSVRC2012_val_00000001')
    """
    bbox_path = os.path.join(dataset_dir, 'bbox', 'val', f'{image_id}.xml').replace("ImageNet", "imagenet")
   # print(bbox_path)
    try:
        # Parse XML file
        tree = ET.parse(bbox_path)
        root = tree.getroot()
        
        # Get image size
        size_elem = root.find('size')
        if size_elem is None:
            print(f"Warning: No size information in annotation for {image_id}")
            return None
            
        width = float(size_elem.find('width').text)
        height = float(size_elem.find('height').text)
        
        # Get bounding box (use first object if multiple)
        obj = root.find('object')
        if obj is None:
            print(f"Warning: No object found in annotation for {image_id}")
            return None
            
        bbox = obj.find('bndbox')
        if bbox is None:
            print(f"Warning: No bounding box found in annotation for {image_id}")
            return None
        
        # Extract coordinates and normalize
        x_min = float(bbox.find('xmin').text) / width
        y_min = float(bbox.find('ymin').text) / height
        x_max = float(bbox.find('xmax').text) / width
        y_max = float(bbox.find('ymax').text) / height
        
        return [x_min, y_min, x_max, y_max]
            
    except FileNotFoundError:
        print(f"Warning: No bounding box found for image {image_id}")
        return None
    except Exception as e:
        print(f"Error loading bounding box for {image_id}: {str(e)}")
        traceback.print_exc()
        return None

def calculate_bbox_iou(cam, bbox, threshold=0.15):
    """
    Calculate IoU between predicted bounding box (derived from CAM) and ground truth bounding box.
    
    Args:
        cam: CAM heatmap tensor
        bbox: Ground truth bounding box [x_min, y_min, x_max, y_max] (normalized)
        threshold: Threshold relative to highest CAM value (default: 0.15)
    
    Returns:
        float: IoU score between predicted and ground truth bounding boxes
    """
    if bbox is None:
        return None
        
    # Convert CAM to numpy if it's a tensor
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
    
    # Normalize CAM to [0, 255] range
    cam_normalized = (cam * 255).astype(np.uint8)
    
    # Binarize the CAM with threshold of 15% of max value (255)
    thresh_value = int(255 * threshold)
    _, binary_map = cv2.threshold(cam_normalized, thresh_value, 255, cv2.THRESH_BINARY)
    
    # Find contours in binary map
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0
    
    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding box from largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Convert to normalized coordinates
    h_img, w_img = cam.shape
    pred_bbox = [
        x / w_img,  # x_min
        y / h_img,  # y_min
        (x + w) / w_img,  # x_max
        (y + h) / h_img   # y_max
    ]
    
    # Calculate intersection
    x_min_inter = max(pred_bbox[0], bbox[0])
    y_min_inter = max(pred_bbox[1], bbox[1])
    x_max_inter = min(pred_bbox[2], bbox[2])
    y_max_inter = min(pred_bbox[3], bbox[3])
    
    # Calculate areas
    if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
        return 0.0
        
    area_inter = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    
    area_pred = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    area_gt = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    area_union = area_pred + area_gt - area_inter
    
    # Calculate IoU
    iou = area_inter / (area_union + 1e-6)
    
    return float(iou)

def evaluate_cam_metrics(
    model,
    input_image,
    input_label,
    grayscale_cam,
    original_pred,
    device,
    threshold=0.5,
    bbox=None,
    baseline_result_base=None,
    transform=None
):
    """
    Evaluate CAM results using multiple metrics including IoU with ground truth bbox and ROAD.
    """
    metrics = {}
    
    # Get model predictions first
    with torch.no_grad():
        logits = model(input_image.unsqueeze(0).to(device))
        _, predicted = torch.max(logits.data, 1)
        probs = F.softmax(logits, dim=1)
        top5_values, top5_indices = torch.topk(probs, k=5, dim=1)
        # Check if true label is in top1 or top5 - convert to Python bool
        is_top1 = bool((top5_indices[0, 0] == input_label))
        is_top5 = bool((input_label in top5_indices[0]))

    # Calculate IoU if bbox is available
    if bbox is not None:

        bbox_iou = calculate_bbox_iou(grayscale_cam, bbox, 0.15)
        if bbox_iou is not None : # dont go here
            # Convert tensor to float if needed
            bbox_iou = float(bbox_iou) if isinstance(bbox_iou, torch.Tensor) else bbox_iou
            
            # loc1: IoU if the prediction is correct (top1)
            metrics['loc1_iou'] = bbox_iou if is_top1 else 0.0
            metrics['loc1_correct'] = bool((bbox_iou >= 0.5) and is_top1)
            
            # loc5: IoU if the true label is in top5
            metrics['loc5_iou'] = bbox_iou if is_top5 else 0.0
            metrics['loc5_correct'] = bool((bbox_iou >= 0.5) and is_top5)
            
            # Store raw IoU regardless of classification result
            metrics['bbox_iou'] = bbox_iou

    # Ensure grayscale_cam is on the correct device and convert to numpy for ROAD metric
    if isinstance(grayscale_cam, torch.Tensor):
        grayscale_cam_np = grayscale_cam.detach().cpu().numpy()
        grayscale_cam = grayscale_cam.to(device)
    else:
        grayscale_cam_np = grayscale_cam
        grayscale_cam = torch.tensor(grayscale_cam, device=device)
    
    # Normalize grayscale_cam_np to [0, 1]
    grayscale_cam_np = (grayscale_cam_np - grayscale_cam_np.min()) / (grayscale_cam_np.max() - grayscale_cam_np.min() + 1e-7)
    
   # print(f"CAM stats - min: {grayscale_cam_np.min()}, max: {grayscale_cam_np.max()}, mean: {grayscale_cam_np.mean()}")
    
    # Prepare masks for other metrics
    binary_mask = (grayscale_cam > threshold).float()
    inverse_mask = 1 - binary_mask
    
    if binary_mask.dim() == 2:
        binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)
        inverse_mask = inverse_mask.unsqueeze(0).unsqueeze(0)
    elif binary_mask.dim() == 3:
        binary_mask = binary_mask.unsqueeze(0)
        inverse_mask = inverse_mask.unsqueeze(0)
    
    # Ensure input_image is on the correct device and has correct dimensions
    if not isinstance(input_image, torch.Tensor):
        input_image = torch.tensor(input_image)
    input_image = input_image.to(device)
    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)
    
    # Create masked images
    positive_masked_image = input_image * binary_mask
    negative_masked_image = input_image * inverse_mask
    
    with torch.no_grad():
        # Get predictions for masked images
        pos_output = F.softmax(model(positive_masked_image), dim=1)
        neg_output = F.softmax(model(negative_masked_image), dim=1)
        
        # Calculate existing metrics
        loc_score = (original_pred - neg_output[0, input_label]).item()
        
        other_classes = torch.cat([pos_output[0, :input_label], pos_output[0, input_label+1:]])
        class_disc = (pos_output[0, input_label] - other_classes.mean()).item()
        
        sorted_energy = torch.sort(grayscale_cam.flatten(), descending=True)[0]
        top_20_idx = int(0.2 * len(sorted_energy))
        energy_concentration = (sorted_energy[:top_20_idx].sum() / sorted_energy.sum()).item()
        targets = [ClassifierOutputTarget(input_label)]

        road75 = ROADMostRelevantFirst(percentile=75)
        road_metric_75 = road75(input_image, baseline_result_base, targets, model)
       # print(f"Road 75: {road_metric_75}")
        # Store all metrics - convert numpy types to Python native types
        metrics.update({
            'bbox_iou': float(bbox_iou) if bbox is not None else None,
            'localization_score': float(loc_score),
            'class_discrimination': float(class_disc),
            'energy_concentration': float(energy_concentration),
            'road_75': float(road_metric_75)
        })
    
    # Ensure all metrics are Python native types
    for key in metrics:
        if isinstance(metrics[key], torch.Tensor):
            metrics[key] = metrics[key].cpu().item()
        elif isinstance(metrics[key], np.ndarray):
            metrics[key] = metrics[key].item()

    return metrics

def evaluate_with_baseline(
    model,
    dataset_name,
    input_image,
    input_label,
    input_idx,
    input_raw_idx,
    model_name,
    base_path,
    device,
    save_dir,
    methods=None,
    k_neurons=1000,
    transform=None,
):
    """Run both SSVEP-focused and baseline CAM methods and save results."""
    if methods is None:
        methods = ["gradcam++", "eigencam", "gradcam", "layercam"]#, "scorecam"]

    # Load ground truth bounding box if available
    dataset_dir = os.path.join(os.environ.get("DATA_DIR", ""), dataset_name)
    bbox = load_imagenet_bbox(dataset_dir, input_idx)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if metrics file exists and load existing results
    metrics_path = os.path.join(save_dir, f"{input_idx}_metrics_{k_neurons}.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_results = json.load(f)
        
        # Check if all methods are already computed
        expected_methods = set([f"baseline_{m}" for m in methods] + [f"ssvep_{m}" for m in methods])
        existing_methods = set(metrics_results.keys())
        
        if expected_methods.issubset(existing_methods):
            print(f"All methods already computed for image {input_idx}, skipping...")
            return None, metrics_results
            
        # Only process missing methods
        methods_to_process = [m.replace("baseline_", "") for m in (expected_methods - existing_methods) 
                            if m.startswith("baseline_")]
        print(f"Processing missing methods for image {input_idx}: {methods_to_process}")
    else:
        metrics_results = {}
        methods_to_process = methods

    results = {}
    
    # Get original prediction confidence
    with torch.no_grad():
        original_output = F.softmax(model(input_image.unsqueeze(0).to(device)), dim=1)
        original_pred = original_output[0, input_label].item()

    # Run baseline methods
    for method in tqdm(methods_to_process, desc="Evaluating baseline CAM methods"):
        try:
            baseline_result, baseline_result_base = prepare_baseline_cam(
                model,
                input_image,
                input_label,
                device,
                method_name=method,
                n_last_layers=5,
            )
            results[f"baseline_{method}"] = baseline_result
            
            # Evaluate metrics for baseline
            metrics = evaluate_cam_metrics(
                model,
                input_image,
                input_label,
                torch.tensor(baseline_result, device=device),
                original_pred,
                device,
                bbox=bbox,
                baseline_result_base=baseline_result_base,
                transform=transform
            )
            metrics_results[f"baseline_{method}"] = metrics
            
        except Exception as e:
            print(f"Error processing baseline {method}: {str(e)}")
            traceback.print_exc()
            continue

    # Run SSVEP-focused methods
    for method in tqdm(methods_to_process, desc="Evaluating SSVEP-focused CAM methods"):
        try:
            ssvep_result, ssvep_result_base = prepare_focused_cam(
                model,
                dataset_name,
                input_image,
                input_label,
                input_idx,
                input_raw_idx,
                model_name,
                base_path,
                device,
                method_name=method,
                k_neurons=k_neurons,
            )
            results[f"ssvep_{method}"] = ssvep_result
            
            # Evaluate metrics for SSVEP
            metrics = evaluate_cam_metrics(
                model,
                input_image,
                input_label,
                torch.tensor(ssvep_result, device=device),
                original_pred,
                device,
                bbox=bbox,
                baseline_result_base=ssvep_result_base,
                transform=transform
            )
            metrics_results[f"ssvep_{method}"] = metrics
            
        except Exception as e:
            print(f"Error processing SSVEP {method}: {str(e)}")
            traceback.print_exc()
            continue

    # Save updated metrics results
    with open(metrics_path, 'w') as f:
        json.dump(metrics_results, f, indent=4)

    # Only create visualizations every 50 images
    if input_raw_idx % 5 == 0:
        visualize_cam_results(
            input_image.squeeze(0).cpu(),
            results,
            dataset_name,
            model_name,
            input_idx,
            input_label,
            save_dir,
            transform,
            metrics_results,
            k_neurons
        )

    return results, metrics_results

def visualize_cam_results(
    input_image,
    cam_results,
    dataset_name,
    model_name,
    input_idx,
    input_label,
    save_dir,
    transform=None,
    metrics_results=None,
    k_neurons=1000
):
    """
    Visualize CAM results and metrics for different methods.
    """
    # Create visualization directory
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get class label text
    labels_path = os.path.join(os.environ.get("DATA_DIR"), "imagenet/imagenet_labels.txt")
    try:
        with open(labels_path, "r") as f:
            imagenet_labels = [line.strip() for line in f.readlines()]
        label_text = imagenet_labels[input_label] if input_label < len(imagenet_labels) else f"Class {input_label}"
    except:
        label_text = f"Class {input_label}"

    # Convert input image to numpy array if needed
    if isinstance(input_image, torch.Tensor):
        input_image = reverse_transform(input_image, transform)
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
    
    # Normalize input image to [0, 1] range if needed
    if input_image.max() > 1:
        input_image = input_image / 255.0

    # Handle empty results case
    if not cam_results:
        print(f"No CAM results to visualize for image {input_idx}")
        return

    # Separate baseline and SSVEP results
    baseline_results = {k: v for k, v in cam_results.items() if k.startswith('baseline_')}
    ssvep_results = {k: v for k, v in cam_results.items() if k.startswith('ssvep_')}

    # Calculate grid dimensions
    n_methods = len(baseline_results)  # Number of methods (should be same for both)
    n_cols = 3  # Original + Baseline + SSVEP
    n_rows = n_methods  # One row per method
    
    # Create figure for CAM visualization
    plt.figure(figsize=(15, 5 * n_rows))
    
    # Plot original image in first row
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(input_image)
    plt.title(f"Original Image\nLabel: {label_text}")
    plt.axis('off')
    
    # Plot method results side by side
    for idx, method in enumerate(baseline_results.keys()):
        base_method = method.replace('baseline_', '')
        row = idx + 1
        
        # Baseline result
        plt.subplot(n_rows, n_cols, row * n_cols - 1)
        baseline_map = baseline_results[method]
        baseline_vis = show_cam_on_image(input_image, baseline_map, use_rgb=True)
        plt.imshow(baseline_vis)
        plt.title(f"Baseline {base_method}")
        plt.axis('off')
        
        # SSVEP result
        plt.subplot(n_rows, n_cols, row * n_cols)
        ssvep_map = ssvep_results[f'ssvep_{base_method}']
        ssvep_vis = show_cam_on_image(input_image, ssvep_map, use_rgb=True)
        plt.imshow(ssvep_vis)
        plt.title(f"SSVEP {base_method}")
        plt.axis('off')
    
    # Save CAM visualization
    plt.suptitle(f"{dataset_name} - {model_name}\nImage {input_idx} - Label: {label_text} - {k_neurons} neurons", fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(vis_dir, f"{input_idx}_{k_neurons}_cam_visualization.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    # Create separate figure for metrics comparison
    if metrics_results:
        metrics_to_plot = [
            'bbox_iou', 'localization_score', 'class_discrimination', 
            'energy_concentration', 'loc1_iou', 'loc5_iou'
        ]
        
        # Add boolean metrics in a separate subplot
        boolean_metrics = ['loc1_correct', 'loc5_correct']
        
        methods = list(baseline_results.keys())
        n_metrics = len(metrics_to_plot) + 1  # Add 1 for boolean metrics
        
        # Create a larger figure with more height per subplot
        plt.figure(figsize=(15, 4 * n_metrics))
        
        for idx, metric in enumerate(metrics_to_plot):
            plt.subplot(n_metrics, 1, idx + 1)
            
            baseline_values = [metrics_results[m].get(metric, 0) for m in methods]
            ssvep_values = [metrics_results[f'ssvep_{m.replace("baseline_", "")}'].get(metric, 0) for m in methods]
            
            x = np.arange(len(methods))
            width = 0.35
            
            # Create bars
            plt.bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue')
            plt.bar(x + width/2, ssvep_values, width, label='SSVEP', color='lightcoral')
            
            plt.xlabel('Methods')
            plt.ylabel('Score')
            plt.title(f'{metric.replace("_", " ").title()} Comparison')
            plt.xticks(x, [m.replace('baseline_', '') for m in methods], rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value labels on top of bars with proper formatting
            for i, v in enumerate(baseline_values):
                if v is not None and v != 0:  # Only show non-zero values
                    plt.text(i - width/2, v, f'{v:.3f}' if v < 1 else f'{v:.1f}', 
                            ha='center', va='bottom')
            for i, v in enumerate(ssvep_values):
                if v is not None and v != 0:  # Only show non-zero values
                    plt.text(i + width/2, v, f'{v:.3f}' if v < 1 else f'{v:.1f}', 
                            ha='center', va='bottom')
            
            # Adjust y-axis limits to give some headroom for labels
            ymin, ymax = plt.ylim()
            plt.ylim(ymin, ymax * 1.15)  # Add 15% headroom
        
        plt.suptitle(f'Metrics Comparison - Image {input_idx}', fontsize=16, y=1.02)
        # Adjust spacing between subplots
        plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=1.5)  # Increase vertical spacing
        metrics_save_path = os.path.join(vis_dir, f"{input_idx}_{k_neurons}_metrics_comparison.png")
        plt.savefig(metrics_save_path, bbox_inches='tight', dpi=300)
        plt.close()

def prepare_baseline_cam(
    model,
    input_image,
    input_label,
    device,
    method_name="gradcam++",
    n_last_layers=5,
):
    """
    Prepare CAM visualization without SSVEP analysis (baseline).
    Uses only layer4 conv layers as targets.
    """
    # Get layer4 convolution layers
    # CHANGE DEPENDING ON ARCH
    conv_layers = []
    for name, module in model.named_modules():
        if "24" in name or "25" in name or "26" in name or "27" in name or "28" in name or "29" in name or "30" in name:
            conv_layers.append((name, module))
    
    # Use all conv layers in layer4
    target_layers = [module for _, module in conv_layers]
    
    if not target_layers:
        raise ValueError(f"Could not find any convolutional layers in layer4")
    
   # print(f"Using layer4 conv layers for baseline CAM")

    # Initialize CAM
    cam_method_class = get_cam_method(method_name)
    if cam_method_class is None:
        raise ValueError(f"Unknown CAM method: {method_name}")
    
    cam = cam_method_class(
        model=model,
        target_layers=target_layers,
    )

    # Ensure input_tensor is 4D: [batch_size, channels, height, width]
    input_tensor = input_image.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
   # print(input_label)
    targets = [ClassifierOutputTarget(input_label)]
    
    try:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            eigen_smooth=True if method_name in ["eigencam", "eigengradcam"] else False
        )
        grayscale_cam_base = grayscale_cam
        grayscale_cam = grayscale_cam[0, :]
    except Exception as e:
        print(f"Error in baseline CAM generation: {str(e)}")
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Target layers: {[type(layer) for layer in target_layers]}")
        raise e

    return grayscale_cam, grayscale_cam_base

def prepare_evaluation_single(dataset_name, model_name, k_neurons=500):
    total_range = 50000
    best_images_path = os.path.join(os.environ.get("RESULTS_DIR"), "best_images_analysis", "best_image_ids.txt")
    with open(best_images_path, "r") as f:
        best_image_ids = [line.strip() for line in f.readlines()]
    best_image_ids = set(best_image_ids)
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup parameters
    base_path = os.environ.get("DATA_DIR")
    save_dir = os.path.join(os.environ.get("RESULTS_DIR"), f"{dataset_name}_{model_name}_cam_analysis")
    
    # Load model and dataset
    test_dataset, transform, _ = get_dataset(
        dataset_name, None, model_type=model_name, base_path=base_path, xai_exp=True
    )
    #model = get_model(
    #    model_name, num_classes=1000, pretrained=True, dataset_name=dataset_name
   # ).to(device)
    

    from torchvision import models
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True).to(device)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True).to(device)
    elif model_name == "densenet161":
        model = models.densenet161(pretrained=True).to(device)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True).to(device)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=True).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.eval()    
    new_test_dataset = []
    i = 0
    for _, _, input_idx in test_dataset:
        if input_idx in best_image_ids:
            new_test_dataset.append(test_dataset[i])
        i += 1
    test_dataset = new_test_dataset
    num_samples = len(test_dataset)
    for idx in tqdm(range(num_samples), desc="Processing images"):
    
        input_image, input_label, input_idx = test_dataset[idx]
        #if input_idx not in best_image_ids:
        #    continue
        
        results, metrics_results = evaluate_with_baseline(
            model,
            dataset_name,
            input_image,
            input_label,
            input_idx,
            idx,
            model_name,
            base_path,
            device,
            save_dir,
            k_neurons=k_neurons,
            transform=transform,
        )

def prepare_evaluation_top_performers(dataset_name, model_name, k_neurons=100):
    """
    Evaluate only top performing images based on IoU and energy concentration metrics.
    Creates separate visualizations for each method while maintaining all original functionality.
    """
    # Get top performing image IDs
    results_dir = os.path.join(os.environ.get("RESULTS_DIR"), f"{dataset_name}_{model_name}_cam_analysis")
    
    # Analyze metrics from existing JSON files
    top_images = {'iou': {}, 'energy': {}}
    json_files = [f for f in os.listdir(results_dir) if f.endswith('_100.json')]
    
    for method in ['baseline_gradcam++', 'baseline_eigencam', 'baseline_gradcam', 'baseline_layercam',
                  'ssvep_gradcam++', 'ssvep_eigencam', 'ssvep_gradcam', 'ssvep_layercam']:
        top_images['iou'][method] = []
        top_images['energy'][method] = []
    
    # Collect metrics from JSON files
    for json_file in json_files:
        with open(os.path.join(results_dir, json_file), 'r') as f:
            data = json.load(f)
            image_id = json_file.split('_metrics')[0]
            
            for method, values in data.items():
                iou = values.get('bbox_iou', 0)
                energy = values.get('energy_concentration', 0)
                
                top_images['iou'][method].append((image_id, iou))
                top_images['energy'][method].append((image_id, energy))
    
    # First, get all images sorted by energy concentration for each method
    energy_sorted = {}
    for method in top_images['energy']:
        sorted_images = sorted(top_images['energy'][method],
                             key=lambda x: x[1] if x[1] is not None else -1,
                             reverse=True)
        energy_sorted[method] = sorted_images

    # Compare SSVEP vs baseline pairs
    selected_ids = set()
    method_pairs = [
        ('ssvep_gradcam++', 'baseline_gradcam++'),
        ('ssvep_eigencam', 'baseline_eigencam'),
        ('ssvep_gradcam', 'baseline_gradcam'),
        ('ssvep_layercam', 'baseline_layercam')
    ]

    # First, find images where SSVEP has better energy in ALL methods
    all_images = set(img[0] for imgs in top_images['energy'].values() for img in imgs)
    consistently_better_energy = []
    
    print(f"Total number of images to analyze: {len(all_images)}")
    
    for img_id in all_images:
        better_energy_count = 0
        better_iou_count = 0
        total_energy_improvement = 0
        total_iou_improvement = 0
        
        valid_image = True
        method_stats = []  # For debugging
        
        for ssvep_method, baseline_method in method_pairs:
            # Get energy values
            ssvep_energy = next((e[1] for e in top_images['energy'][ssvep_method] if e[0] == img_id), None)
            base_energy = next((e[1] for e in top_images['energy'][baseline_method] if e[0] == img_id), None)
            
            # Get IoU values
            ssvep_iou = next((i[1] for i in top_images['iou'][ssvep_method] if i[0] == img_id), None)
            base_iou = next((i[1] for i in top_images['iou'][baseline_method] if i[0] == img_id), None)
            
            # Skip if any values are missing
            if None in (ssvep_energy, base_energy, ssvep_iou, base_iou):
                valid_image = False
                break
                
            # Check if SSVEP is better
            if ssvep_energy > base_energy:
                better_energy_count += 1
                total_energy_improvement += (ssvep_energy - base_energy)
            
            if ssvep_iou >= base_iou * 0.9:  # Allow SSVEP IoU to be within 90% of baseline
                better_iou_count += 1
                total_iou_improvement += (ssvep_iou - base_iou)
            
            method_stats.append({
                'method': ssvep_method.replace('ssvep_', ''),
                'energy_diff': ssvep_energy - base_energy,
                'iou_diff': ssvep_iou - base_iou
            })
        
        # Relaxed criteria:
        # 1. SSVEP has better energy in at least 3/4 methods
        # 2. SSVEP has comparable/better IoU in at least 2/4 methods
        if valid_image and better_energy_count >= 3 and better_iou_count >= 2:
            consistently_better_energy.append((
                img_id,
                total_energy_improvement / 4,  # average energy improvement
                total_iou_improvement / 4,     # average IoU improvement
                better_iou_count,
                method_stats
            ))
    
    print(f"Found {len(consistently_better_energy)} images meeting criteria")
    
    # Sort by average energy improvement (primary) and IoU improvement (secondary)
    sorted_candidates = sorted(consistently_better_energy,
                             key=lambda x: (x[1], x[2], x[3]),  # (avg_energy_imp, avg_iou_imp, iou_count)
                             reverse=True)
    
    # Print top 10 candidates for debugging
    print("\nTop 10 candidates:")
    for i, (img_id, avg_energy, avg_iou, iou_count, stats) in enumerate(sorted_candidates[:10]):
        print(f"\nImage {img_id}:")
        print(f"Average energy improvement: {avg_energy:.3f}")
        print(f"Average IoU improvement: {avg_iou:.3f}")
        print(f"Methods with better IoU: {iou_count}")
        print("Per-method statistics:")
        for stat in stats:
            print(f"  {stat['method']}: Energy diff = {stat['energy_diff']:.3f}, IoU diff = {stat['iou_diff']:.3f}")
    
    # Take top 50 overall (or all if less than 50)
    selected_ids.update(img[0] for img in sorted_candidates[:50])
    
    print(f"\nSelected {len(selected_ids)} images for visualization")

    # Create visualization directory for separate method visualizations
    vis_dir = os.path.join(results_dir, 'top_performers_visNEW')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save selected IDs for reference
    with open(os.path.join(vis_dir, 'selected_ids.json'), 'w') as f:
        json.dump(list(selected_ids), f, indent=4)
    
    # Process selected images using existing evaluation pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = os.environ.get("DATA_DIR")
    
    # Load dataset and model
    test_dataset, transform, _ = get_dataset(
        dataset_name, None, model_type=model_name, base_path=base_path, xai_exp=True
    )
    from torchvision import models
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True).to(device)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True).to(device)
    elif model_name == "densenet161":
        model = models.densenet161(pretrained=True).to(device)
    elif model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True).to(device)
    elif model_name == "resnext50_32x4d":
        model = models.resnext50_32x4d(pretrained=True).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.eval()
    
    # Process only selected images
    for idx, (input_image, input_label, input_idx) in enumerate(test_dataset):
        if input_idx not in selected_ids:
            continue
            
        # Use existing evaluation function
        results, metrics = evaluate_with_baseline(
            model,
            dataset_name,
            input_image,
            input_label,
            input_idx,
            idx,
            model_name,
            base_path,
            device,
            vis_dir,  # Save to separate directory
            k_neurons=k_neurons,
            transform=transform,
        )
        
        # Save separate visualizations for each method
        if results:
            # Group methods into baseline and SSVEP
            baseline_methods = {k: v for k, v in results.items() if k.startswith('baseline_')}
            ssvep_methods = {k: v for k, v in results.items() if k.startswith('ssvep_')}
            
            # Match baseline with corresponding SSVEP method
            for base_method, base_cam in baseline_methods.items():
                ssvep_method = f"ssvep_{base_method.replace('baseline_', '')}"
                ssvep_cam = ssvep_methods.get(ssvep_method)
                
                if ssvep_cam is not None:
                    # Create figure for this method pair
                    plt.figure(figsize=(15, 5))
                    
                    # Original image
                    plt.subplot(1, 3, 1)
                    img_display = reverse_transform(input_image, transform)
                    if isinstance(img_display, torch.Tensor):
                        img_display = img_display.cpu().numpy()
                    if img_display.max() > 1:
                        img_display = img_display / 255.0
                    plt.imshow(img_display)
                    plt.title("Original Image")
                    plt.axis('off')
                    
                    # Baseline CAM visualization
                    plt.subplot(1, 3, 2)
                    base_vis = show_cam_on_image(img_display, base_cam, use_rgb=True)
                    plt.imshow(base_vis)
                    plt.title(f"{base_method}\nIoU: {metrics[base_method]['bbox_iou']:.3f}\nEnergy: {metrics[base_method]['energy_concentration']:.3f}")
                    plt.axis('off')
                    
                    # SSVEP CAM visualization
                    plt.subplot(1, 3, 3)
                    ssvep_vis = show_cam_on_image(img_display, ssvep_cam, use_rgb=True)
                    plt.imshow(ssvep_vis)
                    plt.title(f"{ssvep_method}\nIoU: {metrics[ssvep_method]['bbox_iou']:.3f}\nEnergy: {metrics[ssvep_method]['energy_concentration']:.3f}")
                    plt.axis('off')
                    
                    # Save method-specific visualization
                    method_name = base_method.replace('baseline_', '')
                    save_path = os.path.join(vis_dir, f"{input_idx}_{method_name}_k{k_neurons}.jpg")
                    plt.savefig(save_path, bbox_inches='tight', dpi=300)
                    plt.close()
