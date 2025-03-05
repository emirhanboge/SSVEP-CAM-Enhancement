import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Modulator:
    """
    Handles image modulation operations for creating flickering sequences.
    """

    def __init__(self):
        pass

    @staticmethod
    def freqtag(fps, f, t, sin_max=1.0, sin_min=0.5, phi=0.0):
        """
        Generate sinusoidal modulation signal.

        Args:
            fps (float): Frames per second
            f (float): Frequency in Hz
            t (float): Time point
            sin_max (float): Maximum amplitude
            sin_min (float): Minimum amplitude
            phi (float): Phase offset

        Returns:
            float: Modulation value at time t
        """
        this_angle = 2 * np.pi * f * (t / fps) + phi
        this_scaling = (sin_max - sin_min) / 2
        this_sinus = np.sin(this_angle) * this_scaling + this_scaling + sin_min
        return this_sinus

    @staticmethod
    def apply_sin_modulation(image, luminance):
        """
        Apply sinusoidal modulation to entire image.

        Args:
            image (torch.Tensor): Input image tensor
            luminance (float): Luminance scaling factor

        Returns:
            torch.Tensor: Modulated image
        """
        channels, height, width = image.shape
        modulated_image = torch.clone(torch.tensor(image))

        for i in range(height):
            for j in range(width):
                modulated_image[:, i, j] *= luminance

        return modulated_image

    @staticmethod
    def apply_sin_modulation_half(image, luminance, direction):
        """
        Apply sinusoidal modulation to half of the image.

        Args:
            image (torch.Tensor): Input image tensor
            luminance (float): Luminance scaling factor
            direction (str): 'L' for left half, 'R' for right half

        Returns:
            torch.Tensor: Partially modulated image
        """
        channels, height, width = image.shape
        modulated_image = torch.clone(torch.tensor(image))
        half_width = width // 2

        if direction == "L":
            modulated_image[:, :, :half_width] *= luminance
        elif direction == "R":
            modulated_image[:, :, half_width:] *= luminance

        return modulated_image


class FlickerImagesDataset(Dataset):
    """
    Dataset class for handling flickering image sequences.
    """

    def __init__(self, flicker_images, class_names):
        self.flicker_images = flicker_images
        self.class_names = class_names
        self.image_list = flicker_images

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.image_list[idx]


class FlickerImageGenerator:
    """
    Generates sequences of flickering images for neural analysis.
    """

    def __init__(
        self,
        freq=6,
        freq_left=6,
        freq_right=7.5,
        num_flicker_images=60,
        half_width=False,
        class_idx=None,
    ):
        """
        Initialize generator with modulation parameters.

        Args:
            freq (float): Base frequency for full-image modulation
            freq_left (float): Frequency for left half modulation
            freq_right (float): Frequency for right half modulation
            num_flicker_images (int): Number of images in sequence
            half_width (bool): Whether to modulate half images
            class_idx (int, optional): Class index for specific analysis
        """
        self.freq = freq
        self.freq_left = freq_left
        self.freq_right = freq_right
        self.num_flicker_images = num_flicker_images
        self.half_width = half_width
        self.class_idx = class_idx

    def generate_images(self, image, num_images, fps=60):
        generated_images = []
        for i in range(num_images):
            luminance = Modulator.freqtag(
                fps, f=self.freq, t=i, sin_max=1.0, sin_min=0.5, phi=0.0
            )
            modulated_image = Modulator.apply_sin_modulation(image, luminance)
            modulated_image = torch.tensor(modulated_image)
            generated_images.append(modulated_image)

        return generated_images

    def generate_images_half(self, image, num_images, fps=60):
        generated_images = []
        for i in range(num_images):
            luminance = Modulator.freqtag(
                fps, f=self.freq_left, t=i, sin_max=1.0, sin_min=0.5, phi=0.0
            )
            modulated_image = Modulator.apply_sin_modulation_half(image, luminance, "L")
            luminance = Modulator.freqtag(
                fps, f=self.freq_right, t=i, sin_max=1.0, sin_min=0.5, phi=0.0
            )
            modulated_image = Modulator.apply_sin_modulation_half(
                modulated_image, luminance, "R"
            )
            modulated_image = torch.tensor(modulated_image)
            generated_images.append(modulated_image)

        return generated_images

    def create_flicker_images_dataloaders(
        self, flicker_images, batch_size=1, num_workers=os.cpu_count()
    ):
        dataloaders = []
        for flicker_images_part in flicker_images:
            images, labels = zip(*flicker_images_part)
            dataset = FlickerImagesDataset(images, labels)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            dataloaders.append(dataloader)

        return dataloaders

    def create_flicker_images(self, images, labels):
        """
        The number of labels from each class should be the same.
        If not, the number of images from each class will be the number of the class with the least number of images.
        The remaining images from the classes with more images will be ignored.
        """
        if len(images) != len(labels):
            raise ValueError("The number of images and labels should be the same.")
        if self.num_flicker_images < 60:
            raise ValueError("The number of flicker images should be at least 60.")

        labels = [int(label) for label in labels]

        instances_for_each_class = {}
        for idx, label in enumerate(labels):
            if label not in instances_for_each_class:
                instances_for_each_class[label] = [images[idx]]
            else:
                instances_for_each_class[label].append(images[idx])

        instance_counts = [len(instances_for_each_class[label]) for label in labels]
        num_images_to_select_from_each_class = min(instance_counts)

        flicker_images = []
        for label in instances_for_each_class:
            for image in instances_for_each_class[label][
                :num_images_to_select_from_each_class
            ]:
                flicker_images.append([])
                image = np.array(image)
                if self.half_width:
                    generated_images = self.generate_images_half(
                        image, self.num_flicker_images, self.num_flicker_images
                    )
                else:
                    generated_images = self.generate_images(
                        image, self.num_flicker_images, self.num_flicker_images
                    )
                for generated_image in generated_images:
                    flicker_images[-1].append((generated_image, label))
        return flicker_images

    def generate_flickerloaders(self, images, labels):
        images = self.create_flicker_images(images, labels)
        flicker_loaders = self.create_flicker_images_dataloaders(images)

        print(f"Number of flicker loaders: {len(flicker_loaders)}")
        print(f"Number of images in each flicker loader: {len(flicker_loaders[0])}\n")

        return flicker_loaders
