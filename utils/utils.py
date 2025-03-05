import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torchvision import models
from torch.utils.data import Subset
from tqdm import tqdm
from dotenv import load_dotenv

from src.data_prep import prepare_data
from src.get_scores import GetScores

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")
RESULTS_DIR = os.environ.get("RESULTS_DIR")


def convert_keys_to_strings(d):
    if not isinstance(d, dict):
        return d
    return {str(k): convert_keys_to_strings(v) for k, v in d.items()}


def get_top_neurons(
    model,
    selected_images,
    selected_labels,
    device,
    base_path,
    dataset_name,
    model_name,
    class_name=None,
    xai_exp=False,
    ablation_study=False,
    base_input_idx=None,
    input_idx=None,
):
    if xai_exp == True:
        top_neurons_path = f"{RESULTS_DIR}/{dataset_name}_{model_name}_xai_exp/{input_idx}_top_neurons.json"
        if os.path.exists(top_neurons_path):
            with open(top_neurons_path, "r") as f:
                top_neurons = json.load(f)
            return top_neurons
    top_neurons_path = f"{RESULTS_DIR}/{dataset_name}_{model_name}_top_k_neurons.json"
    if class_name:
        top_neurons_path = f"{RESULTS_DIR}/class_specific/{class_name}/{class_name}_{dataset_name}_{model_name}_top_k_neurons.json"
    if not os.path.exists(top_neurons_path) or xai_exp == True:
        get_scores = GetScores(
            model,
            selected_images,
            selected_labels,
            device,
            base_path=base_path,
            class_name=class_name,
            model_name=model_name,
            dataset_name=dataset_name,
            xai_exp=xai_exp,
        )
        top_neurons = get_scores.return_snr_neurons()
        if xai_exp == True:
            os.makedirs(
                f"{RESULTS_DIR}/{dataset_name}_{model_name}_xai_exp", exist_ok=True
            )
            top_neurons = {k: float(v) for k, v in top_neurons.items()}
            with open(
                f"{RESULTS_DIR}/{dataset_name}_{model_name}_xai_exp/{input_idx}_top_neurons.json",
                "w",
            ) as f:
                json.dump(convert_keys_to_strings(top_neurons), f, indent=4)
            with open(
                f"{RESULTS_DIR}/{dataset_name}_{model_name}_xai_exp/{input_idx}_top_neurons.json",
                "r",
            ) as f:
                top_neurons = json.load(f)
            return top_neurons

        os.makedirs("results", exist_ok=True)
        if class_name:
            os.makedirs(f"{RESULTS_DIR}/class_specific", exist_ok=True)
            os.makedirs(f"{RESULTS_DIR}/class_specific/{class_name}", exist_ok=True)
        with open(top_neurons_path, "w") as f:
            json.dump(convert_keys_to_strings(top_neurons), f, indent=4)

    with open(top_neurons_path, "r") as f:
        top_neurons = json.load(f)

    if ablation_study:
        # Invert the selection: keep all neurons except the top ones
        all_neurons = torch.ones_like(top_neurons)
        ablation_mask = all_neurons - top_neurons
        # Now ablation_mask has 1s for neurons we want to keep (non-top neurons)
        # and 0s for top neurons we want to remove
        
        # Apply the ablation mask
        filtered_activations = top_neurons * ablation_mask
    else:
        # Original behavior: only keep top neurons
        filtered_activations = top_neurons

    return filtered_activations


def reverse_transform(transformed_image, transform):
    # Extract mean and std from the transform object
    mean = torch.tensor(
        transform.mean, dtype=transformed_image.dtype, device=transformed_image.device
    )
    std = torch.tensor(
        transform.std, dtype=transformed_image.dtype, device=transformed_image.device
    )

    # Reverse normalization
    transformed_image = transformed_image * std[:, None, None] + mean[:, None, None]

    # Convert to numpy and scale to [0, 255]
    transformed_image = transformed_image.permute(1, 2, 0).cpu().numpy()
    transformed_image = (transformed_image * 255).clip(0, 255).astype("uint8")

    return transformed_image


def parse_neuron_keys(neuron_dict, return_snr=False):
    """Parse neuron keys and optionally return SNR values."""
    parsed_neurons = []
    for key, value in neuron_dict.items():
        # Assuming key format: "('layer_name', filter_idx)"
        layer_name = key.split("'")[1]
        filter_idx = int(key.split(",")[1].strip(" )"))
        if return_snr:
            parsed_neurons.append((layer_name, filter_idx, value))
        else:
            parsed_neurons.append((layer_name, filter_idx))
    
    # Sort by SNR value (descending)
    if return_snr:
        return sorted(parsed_neurons, key=lambda x: x[2], reverse=True)
    return sorted(parsed_neurons, key=lambda x: neuron_dict[f"('{x[0]}', {x[1]})"], reverse=True)


def modify_weights_ablation(model, neurons_to_remove):
    deleted_filter_count = 0
    for (layer_name, neuron_number) in neurons_to_remove:
        layer = dict(model.named_modules())[layer_name]
        with torch.no_grad():
            layer.weight[neuron_number, :, :, :] = 0  # Zero the weights
            if layer.bias is not None:
                layer.bias[neuron_number] = 0
        deleted_filter_count += 1
    return model


def modify_weights(model, neurons_to_remove, k_neurons=10):
    neurons_to_remove = parse_neuron_keys(neurons_to_remove)
    neurons_to_remove = neurons_to_remove[:k_neurons]
    deleted_filter_count = 0
    for (layer_name, neuron_number) in neurons_to_remove:
        layer = dict(model.named_modules())[layer_name]
        with torch.no_grad():
            layer.weight[neuron_number, :, :, :] = 0  # Zero the weights
            if layer.bias is not None:
                layer.bias[neuron_number] = 0
        deleted_filter_count += 1
    return model


def lowest_modify_weights(model, neurons_to_remove, k_neurons=10):
    neurons_to_remove = parse_neuron_keys(neurons_to_remove)
    neurons_to_remove = neurons_to_remove[::-1]
    neurons_to_remove = neurons_to_remove[:k_neurons]
    print(neurons_to_remove)
    deleted_filter_count = 0
    for (layer_name, neuron_number) in neurons_to_remove:
        layer = dict(model.named_modules())[layer_name]
        with torch.no_grad():
            layer.weight[neuron_number, :, :, :] = 0  # Zero the weights
            if layer.bias is not None:
                layer.bias[neuron_number] = 0  # Zero the bias
        deleted_filter_count += 1
    print(f">>> Deleted {deleted_filter_count} filters")
    return model


def randomly_modify_weights(model, neurons_to_remove, k_neurons=10):
    neurons_to_remove = parse_neuron_keys(neurons_to_remove)
    neurons_to_remove = random.sample(
        neurons_to_remove, k_neurons
    )  # Randomly select k neurons
    deleted_filter_count = 0
    for (layer_name, neuron_number) in neurons_to_remove:
        layer = dict(model.named_modules())[layer_name]
        with torch.no_grad():
            layer.weight[neuron_number, :, :, :] = 0  # Zero the weights
            if layer.bias is not None:
                layer.bias[neuron_number] = 0
        deleted_filter_count += 1
    print(f">>> Deleted {deleted_filter_count} filters randomly")
    return model


def load_imagenet_classes():
    with open(f"{DATA_DIR}/imagenet/imagenet_labels.txt", "r") as f:
        lines = f.readlines()
        class_idx_map = {}
        for line in lines:
            class_name = line
            class_name = class_name.strip()  # Remove trailing newline
            class_name = class_name.lower()  # Convert to lowercase
            idx = lines.index(line)
            class_idx_map[class_name] = int(idx)
    return class_idx_map


def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def get_dataset(dataset_name, class_idx, model_type, base_path=DATA_DIR, xai_exp=False):
    (test_loader, selected_images, selected_labels,) = prepare_data(
        dataset_name=dataset_name,
        model_type=model_type,
        base_path=base_path,
        class_idx=class_idx,
        xai_exp=xai_exp,
    )
    return (
        test_loader,
        selected_images,
        selected_labels,
    )


def get_model(
    model_name,
    num_classes,
    pretrained=True,
    dataset_name="ImageNet",
    base_path=DATA_DIR,
):
    # ImageNet Models (Torch Hub)
    if model_name == "inception_v3" and dataset_name == "ImageNet":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "inception_v3", pretrained=pretrained
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50" and dataset_name == "ImageNet":
        model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_resnet50",
            pretrained=pretrained,
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vgg16" and dataset_name == "ImageNet":
        model = torch.hub.load("pytorch/vision:v0.10.0", "vgg16", pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model


def is_img_equal(img1, img2):
    return torch.allclose(img1, img2)


def create_class_specific_dataloader(
    data_loader, target_class_idx, selected_images, img_per_class=50
):
    target_indices = []
    class_dependent_idx = 0
    for i, (img, labels) in enumerate(data_loader.dataset):
        if labels == target_class_idx:
            if not is_img_equal(img, selected_images[class_dependent_idx]):
                target_indices.append(i)
                class_dependent_idx += 1
                if class_dependent_idx == len(selected_images):
                    break

    subset = Subset(data_loader.dataset, target_indices)
    class_specific_loader = torch.utils.data.DataLoader(
        subset,
        batch_size=data_loader.batch_size,
        shuffle=False,
        num_workers=data_loader.num_workers,
        pin_memory=data_loader.pin_memory,
    )
    print(f">>> Number of images in class-specific loader: {len(subset)}")

    return class_specific_loader


def evaluate_model_acc_recall(model, data_loader, device, label_count=1000):
    model.eval()

    accuracy_metric = torchmetrics.Accuracy().to(device)
    recall_metric = torchmetrics.Recall(average="macro", num_classes=label_count).to(
        device
    )

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating model"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            accuracy_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)

    accuracy = accuracy_metric.compute().item() * 100
    recall = recall_metric.compute().item()

    print(
        f"Accuracy of the model on the {len(data_loader.dataset)} test images: {accuracy:.2f}%"
    )
    print(
        f"Recall of the model on the {len(data_loader.dataset)} test images: {recall:.2f}"
    )

    return accuracy, recall


def evaluate_class_specific_accuracy_recall(model, class_specific_loader, device):
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in class_specific_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    class_specific_accuracy = 100 * correct / total
    recall = correct / total

    return class_specific_accuracy, recall
