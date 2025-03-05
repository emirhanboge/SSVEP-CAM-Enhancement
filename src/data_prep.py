import copy
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")
RESULTS_DIR = os.environ.get("RESULTS_DIR")


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.image_ids = []

        self._load_dataset()

        unique_sorted_labels = sorted(list(set(self.labels)))
        print(f">>> Number of classes: {len(unique_sorted_labels)}")

    def _load_dataset(self):
        for class_dir in sorted(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(int(class_dir))
                    self.image_ids.append(os.path.splitext(img_name)[0])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image_id = self.image_ids[idx]
        image = Image.open(img_path).convert("RGB")
        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)
        return image, label, image_id


def load_dataset(dataset_name, transform=None, base_path="data/"):
    os.makedirs(base_path, exist_ok=True)
    if dataset_name == "CIFAR10":
        test_dataset = datasets.CIFAR10(
            root=os.path.join(base_path, f"{dataset_name.lower()}_data"),
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "CIFAR100":
        test_dataset = datasets.CIFAR100(
            root=os.path.join(base_path, f"{dataset_name.lower()}_data"),
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "MNIST":
        test_dataset = datasets.MNIST(
            root=os.path.join(base_path, f"{dataset_name.lower()}_data"),
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "SVHN":
        test_dataset = datasets.SVHN(
            root=os.path.join(base_path, f"{dataset_name.lower()}_data"),
            split="test",
            download=True,
            transform=transform,
        )
    elif dataset_name == "ImageNet":
        val_dir = os.path.join(base_path, "imagenet", "val")
        test_dataset = ImageNetDataset(val_dir, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f">>> {dataset_name} dataset loaded. With {len(test_dataset)} images.")
    return test_dataset


def datasets_exist(dataset_name, base_path=DATA_DIR, class_idx=None):
    if class_idx is not None:
        return os.path.exists(
            os.path.join(
                base_path,
                f"{dataset_name.lower()}_datasets/{dataset_name.lower()}_datasets_{class_idx}",
            )
        )
    return os.path.exists(
        os.path.join(base_path, f"{dataset_name.lower()}_datasets/test.pt")
    )


def load_datasets(dataset_name, base_path=DATA_DIR, class_idx=None):
    if class_idx is not None:
        base_path = os.path.join(
            base_path,
            f"{dataset_name.lower()}_datasets/{dataset_name.lower()}_datasets_{class_idx}",
        )
    else:
        base_path = os.path.join(base_path, f"{dataset_name.lower()}_datasets")
    test_dataloader = torch.load(os.path.join(base_path, "test.pt"))
    with open(os.path.join(base_path, "selected.pkl"), "rb") as f:
        selected_images, selected_labels = pickle.load(f)

    print(f">>> Number of test images: {len(test_dataloader.dataset)}")
    print(f">>> Number of selected images: {len(selected_images)}")
    print(f">>> Number of selected labels: {len(selected_labels)}")

    return test_dataloader, selected_images, selected_labels


def save_datasets(
    test_dataset,
    selected_images,
    selected_labels,
    dataset_name,
    model_extension,
    base_path=DATA_DIR,
    class_idx=None,
):
    if class_idx is not None:
        base_path = os.path.join(
            base_path,
            f"{dataset_name.lower()}_datasets/{dataset_name.lower()}_datasets_{class_idx}",
        )
    else:
        base_path = os.path.join(base_path, f"{dataset_name.lower()}_datasets")

    os.makedirs(base_path, exist_ok=True)
    torch.save(
        test_dataset,
        os.path.join(base_path, f"test.pt"),
    )
    with open(
        os.path.join(base_path, f"selected.pkl"),
        "wb",
    ) as f:
        pickle.dump((selected_images, selected_labels), f)


class SSVEPData(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def create_transform(dataset_name, model_type):
    if model_type.lower() == "inception_v3" and dataset_name.lower() == "imagenet":
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        transform = weights.transforms()
    elif model_type.lower() == "resnet50" and dataset_name.lower() == "imagenet":
        weights = models.ResNet50_Weights.DEFAULT
        transform = weights.transforms()
    elif model_type.lower() == "vgg16" and dataset_name.lower() == "imagenet":
        weights = models.VGG16_Weights.DEFAULT
        transform = weights.transforms()
    elif model_type.lower() == "convnext" and dataset_name.lower() == "imagenet":
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        transform = weights.transforms()
    elif model_type.lower() == "vit" and dataset_name.lower() == "imagenet":
        weights = models.ViT_B_16_Weights.DEFAULT
        transform = weights.transforms()
    elif model_type.lower() == "densenet161" and dataset_name.lower() == "imagenet":
        weights = models.DenseNet161_Weights.DEFAULT
        transform = weights.transforms()
    elif model_type.lower() == "resnext50_32x4d" and dataset_name.lower() == "imagenet":
        weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        transform = weights.transforms()
    elif dataset_name.lower() in ["cifar10", "cifar100"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]
                ),  # CIFAR Normalization
            ]
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f">>> Transform: {transform}")
    return transform


def prepare_data(
    k=100,
    dataset_name="ImageNet",
    model_type="inception_v3",
    base_path=DATA_DIR,
    class_idx=None,
    xai_exp=False,
):
    if datasets_exist(dataset_name, base_path=base_path, class_idx=class_idx):
        print(f"Loading {dataset_name} dataset from {base_path}")
        return load_datasets(dataset_name, base_path=base_path, class_idx=class_idx)

    print(f"Creating {dataset_name} dataset")

    dataset_name = dataset_name.split("_")[0]
    transform = create_transform(dataset_name, model_type)
    test_dataset = load_dataset(dataset_name, transform=transform, base_path=base_path)
    if xai_exp:
        return test_dataset, transform, None

    if os.path.exists(
        os.path.join(
            base_path, f"{dataset_name.lower()}_datasets", "label_img_count.pkl"
        )
    ):
        with open(
            os.path.join(
                base_path, f"{dataset_name.lower()}_datasets", "label_img_count.pkl"
            ),
            "rb",
        ) as f:
            label_img_count = pickle.load(f)
        print(f"Number of images per class: {label_img_count}")
    else:
        label_img_count = {}
        for _, label in tqdm(test_dataset, total=len(test_dataset)):
            if label not in label_img_count:
                label_img_count[label] = 0
            label_img_count[label] += 1
        os.makedirs(
            os.path.join(base_path, f"{dataset_name.lower()}_datasets"), exist_ok=True
        )
        with open(
            os.path.join(
                base_path, f"{dataset_name.lower()}_datasets", "label_img_count.pkl"
            ),
            "wb",
        ) as f:
            pickle.dump(label_img_count, f)
        print(f"Number of images per class: {label_img_count}")

    if class_idx is None:
        print(f"Selecting {k} random images from the test set")
        test_subset_indices = np.random.choice(len(test_dataset), k, replace=False)
        test_subset = [test_dataset[i] for i in test_subset_indices]
        selected_images = [img for img, _ in test_subset]
        selected_labels = [label for _, label in test_subset]
        while len(set(selected_labels)) != k:
            # Group dataset by label
            label_to_indices = {}
            for idx, (_, label) in enumerate(test_dataset):
                if label not in label_to_indices:
                    label_to_indices[label] = []
                label_to_indices[label].append(idx)

            test_subset_indices = np.random.choice(len(test_dataset), k, replace=False)
            test_subset = [test_dataset[i] for i in test_subset_indices]

            selected_indices = []
            selected_images = [img for img, _ in test_subset]
            selected_labels = [label for _, label in test_subset]

            if dataset_name == "CIFAR10":
                for label, indices in label_to_indices.items():
                    selected_indices.extend(
                        np.random.choice(indices, 1, replace=False)
                    )  # 10 per class
                test_subset = [test_dataset[i] for i in selected_indices]
                selected_images = [img for img, _ in test_subset]
                selected_labels = [label for _, label in test_subset]
            elif dataset_name == "CIFAR100":
                for label, indices in label_to_indices.items():
                    selected_indices.extend(
                        np.random.choice(indices, 1, replace=False)
                    )  # 1 per class
                test_subset = [test_dataset[i] for i in selected_indices]
                selected_images = [img for img, _ in test_subset]

            nums = {
                label: selected_labels.count(label) for label in set(selected_labels)
            }
            if (
                dataset_name == "CIFAR10"
                and len(set(selected_labels)) == 10
                and len(selected_images) == 10
                and all([num == 1 for num in nums.values()])
            ):
                print(
                    f"Selected {len(selected_images)} images from {len(set(selected_labels))} classes"
                )
                print(f"Number of images per class: {nums}")
                break
            elif (
                dataset_name == "CIFAR100"
                and len(set(selected_labels)) == 100
                and len(selected_images) == 100
                and all([num == 1 for num in nums.values()])
            ):
                print(
                    f"Selected {len(selected_images)} images from {len(set(selected_labels))} classes"
                )
                print(f"Number of images per class: {nums}")
                break
        if dataset_name == "ImageNet":
            selected_images = [transform(image) for image in selected_images]
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        save_datasets(
            test_loader,
            selected_images,
            selected_labels,
            dataset_name,
            "",
            class_idx=class_idx,
        )
        print(
            f"Saved at {os.path.join(base_path, f'{dataset_name.lower()}_datasets', f'{dataset_name.lower()}_datasets')}"
        )
    else:  # class_idx not None so we need to select images from a specific class only
        # If the experiment is with a class_idx these examples are removed from the test set
        selected_images = []
        selected_labels = []
        copy_test_dataset = copy.deepcopy(test_dataset)
        for i, (img, label) in tqdm(enumerate(test_dataset), total=len(test_dataset)):
            if os.path.exists(
                os.path.join(
                    base_path,
                    f"{dataset_name.lower()}_datasets",
                    f"{dataset_name.lower()}_datasets_{label}",
                )
            ):
                print(f"Class {label} already processed")
                continue
            selected_images.append(transform(img))
            selected_labels.append(label)
            test_dataset.image_paths.pop(i)
            test_dataset.labels.pop(i)
            if (
                len(selected_images) == label_img_count[label] // 2
            ):  # Select half of the images from the class
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
                selected_labels = [int(label) for label in selected_labels]
                save_datasets(
                    test_loader,
                    selected_images,
                    selected_labels,
                    dataset_name,
                    "",
                    class_idx=label,
                )
                selected_images = []  # Reset the selected images
                selected_labels = []  # Reset the selected labels
                test_dataset = copy.deepcopy(
                    copy_test_dataset
                )  # Reset the test_dataset

                print(
                    f"Class {label} done and saved at {os.path.join(base_path, f'{dataset_name.lower()}_datasets', f'{dataset_name.lower()}_datasets_{label}')}"
                )

    test_loader, selected_images, selected_labels = load_datasets(
        dataset_name, base_path=base_path, class_idx=class_idx
    )

    return test_loader, selected_images, selected_labels
