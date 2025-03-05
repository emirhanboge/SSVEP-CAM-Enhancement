import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from tqdm import tqdm

from src.flicker_images import FlickerImageGenerator

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR")
RESULTS_DIR = os.environ.get("RESULTS_DIR")


class GetScores(nn.Module):
    def __init__(
        self,
        model,
        selected_images,
        selected_labels,
        device=torch.device("cpu"),
        freq=6,
        freq_left=6,
        freq_right=7.5,
        half_width=True,
        num_flicker_images=120,
        dataset_name="ImageNet",
        base_path=DATA_DIR,
        class_idx=None,
        class_name=None,
        model_name=None,
        xai_exp=False,
    ):
        """
        Equal number of images from each class should be selected.
        """
        super(GetScores, self).__init__()
        self.model = model
        self.layer_outputs = {}
        self.hooks = []
        self.device = device
        self.half_width = half_width
        self.freq = freq
        self.num_flicker_images = num_flicker_images
        self.class_idx = class_idx
        self.class_specific_neurons = None
        self.class_name = class_name
        self.model_name = model_name

        self.dataset_name = dataset_name
        self.base_path = base_path

        self.images = selected_images
        self.labels = selected_labels

        if class_idx == None and self.flickerloader_exists() and xai_exp == False:
            self.load_flickerloaders()
        else:
            self.flicker_image_generator = FlickerImageGenerator(
                freq=freq,
                freq_left=freq_left,
                freq_right=freq_right,
                half_width=half_width,
                num_flicker_images=num_flicker_images,
                class_idx=class_idx,
            )
            self.flicker_loaders = self.flicker_image_generator.generate_flickerloaders(
                self.images, self.labels
            )
            if class_idx == None and class_name == None and xai_exp == False:
                self.save_flickerloaders()
        self.plotted = False

        self.layer_datas = self.get_neuron_activations()

        self.snr_neurons = self.get_snr_neurons()
        self.snr_neurons = self.sort_neuron_importance(self.snr_neurons)

        # self.plot_fft_top_k_neurons()

    def save_flickerloaders(self):
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "flicker_loaders"), exist_ok=True)
        path = (
            os.path.join(self.base_path, "flicker_loaders")
            + f"_{self.dataset_name}_{self.freq}Hz_is2tag_{self.half_width}_{self.num_flicker_images}fps{self.class_name}.pt"
        )
        torch.save(self.flicker_loaders, path)

    def load_flickerloaders(self):
        path = (
            os.path.join(self.base_path, "flicker_loaders")
            + f"_{self.dataset_name}_{self.freq}Hz_is2tag_{self.half_width}_{self.num_flicker_images}fps{self.class_name}.pt"
        )
        self.flicker_loaders = torch.load(path)
        print(f"Number of flicker loaders: {len(self.flicker_loaders)}")

    def flickerloader_exists(self):
        path = (
            os.path.join(self.base_path, "flicker_loaders")
            + f"_{self.dataset_name}_{self.freq}Hz_is2tag_{self.half_width}_{self.num_flicker_images}fps{self.class_name}.pt"
        )
        return os.path.exists(path)

    def sort_neuron_importance(self, neurons):
        top_neurons = {}
        for layer in neurons:
            for neuron in neurons[layer]:
                snr = neurons[layer][neuron]
                top_neurons[(layer, neuron)] = snr

        top_neurons = dict(
            sorted(top_neurons.items(), key=lambda item: item[1], reverse=True)
        )

        return top_neurons

    def plot_fft_top_k_neurons(self, k=10):
        top_neurons = dict(list(self.snr_neurons.items())[-k:])

        results = {"low_k_neurons": {}, "top_k_neurons": {}}

        plt.figure()
        for loader in self.layer_datas:
            for layer in self.layer_datas[loader]:
                for neuron in self.layer_datas[loader][layer]:
                    if (layer, neuron) in top_neurons:
                        print(f"Layer: {layer}, Neuron: {neuron}")
                        fft_output = np.fft.rfft(
                            self.layer_datas[loader][layer][neuron],
                            n=self.num_flicker_images,
                        )
                        fft_output = np.abs(fft_output)
                        fft_output = fft_output[1:]  # Remove the DC component

                        results["low_k_neurons"][str((layer, neuron))] = {
                            str(i): fft_output[i] for i in range(len(fft_output))
                        }

                        plt.plot(fft_output, label=f"Layer: {layer}, Neuron: {neuron}")

        x_max = len(fft_output)
        x_ticks_6 = np.arange(
            5, x_max, 6
        )  # Adjust starting point to align with FFT output
        x_ticks_7_5 = np.arange(
            6.5, x_max, 7.5
        )  # Adjust starting point to align with FFT output
        x_ticks = np.unique(np.concatenate((x_ticks_6, x_ticks_7_5)))

        plt.xticks(
            x_ticks,
            [f"{x+1:.1f}" if x % 1 != 0 else f"{int(x)+1}" for x in x_ticks],
            rotation=45,
        )
        plt.xlim(1, x_max)

        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.title("FFT of Low K Neurons")
        plt.legend()

        os.makedirs(f"{RESULTS_DIR}/fft_plots", exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/fft_plots/fft_least_k_neurons.png")
        plt.close()

        top_neurons = dict(list(self.snr_neurons.items())[:k])

        plt.figure()
        for loader in self.layer_datas:
            for layer in self.layer_datas[loader]:
                for neuron in self.layer_datas[loader][layer]:
                    if (layer, neuron) in top_neurons:
                        print(f"Layer: {layer}, Neuron: {neuron}")
                        print(len(self.layer_datas[loader][layer][neuron]))
                        fft_output = np.fft.rfft(
                            self.layer_datas[loader][layer][neuron],
                            n=self.num_flicker_images,
                        )
                        print(len(fft_output))
                        fft_output = np.abs(fft_output)
                        fft_output = fft_output[1:]  # Remove the DC component

                        results["top_k_neurons"][str((layer, neuron))] = {
                            str(i): fft_output[i] for i in range(len(fft_output))
                        }

                        plt.plot(fft_output, label=f"Layer: {layer}, Neuron: {neuron}")

        x_max = len(fft_output)
        x_ticks_6 = np.arange(
            5, x_max, 6
        )  # Adjust starting point to align with FFT output
        x_ticks_7_5 = np.arange(
            6.5, x_max, 7.5
        )  # Adjust starting point to align with FFT output
        x_ticks = np.unique(np.concatenate((x_ticks_6, x_ticks_7_5)))

        plt.xticks(
            x_ticks,
            [f"{x+1:.1f}" if x % 1 != 0 else f"{int(x)+1}" for x in x_ticks],
            rotation=45,
        )
        plt.xlim(1, x_max)

        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.title("FFT of Top K Neurons")
        plt.legend()

        os.makedirs(f"{RESULTS_DIR}/fft_plots", exist_ok=True)
        plt.savefig(f"{RESULTS_DIR}/fft_plots/fft_top_k_neurons.png")
        plt.close()

        with open(f"{RESULTS_DIR}/fft_plots/fft_results.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

    def register_hooks(self):
        """
        Inception_v3: 17216 Filters in total
        """
        self.layer_outputs.clear()
        self.hooks.clear()

        aux_layers = ["AuxLogits"]
        if isinstance(self.model, torch.nn.Module):
            cnn_layer_count = 0
            total_filters = 0
            for i, (name, module) in enumerate(self.model.named_modules()):
                # Inception, ResNet, ConNeXt, VGG
                if isinstance(module, nn.Conv2d) and not any(
                    aux_layer in name for aux_layer in aux_layers
                ):
                    if self.model_name == "resnet50" and "downsample" in name:
                        continue
                    cnn_layer_count += 1
                    filter_count = module.weight.shape[0]
                    total_filters += filter_count
                    self.layer_outputs[name] = []
                    hook = module.register_forward_hook(
                        self.get_intermediate_output(name)
                    )
                    self.hooks.append(hook)

                # Vision Transformer
                elif isinstance(module, nn.MultiheadAttention):
                    cnn_layer_count += 1
                    filter_count = module.out_proj.weight.shape[0]
                    total_filters += filter_count
                    self.layer_outputs[name] = []
                    hook = module.register_forward_hook(
                        self.get_intermediate_output(name)
                    )
                    self.hooks.append(hook)
            # Save the stats
            with open(
                f"{RESULTS_DIR}/{self.dataset_name}_{self.model_name}_stats.txt", "w"
            ) as f:
                f.write(f"Total CNN (Attention) Layers: {cnn_layer_count}\n")
                f.write(f"Total Filters (Attention): {total_filters}\n")
        else:
            raise ValueError(
                "Model is not an instance of torch.nn.Module",
                type(self.model),
                self.model,
            )

    def get_intermediate_output(self, layer_name):
        def hook(module, input, output):
            self.layer_outputs[layer_name].append(output)

        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_neuron_activations(self):
        self.register_hooks()

        outputs = {}
        for flicker_loader in self.flicker_loaders:  # tqdm(
            # self.flicker_loaders, desc="Getting Neuron Activations"
            # ):
            layer_datas = {}
            for flicker_images in flicker_loader:
                for i in range(len(flicker_images)):
                    image = flicker_images[i].unsqueeze(0).to(self.device)
                    for name in self.layer_outputs:
                        self.layer_outputs[name] = []
                    with torch.no_grad():
                        _ = self.model(image)
                    for layer in self.layer_outputs:
                        if layer not in layer_datas:
                            layer_datas[layer] = {}
                        if self.model_name == "vit":
                            means = (  # Calculate the mean of the activations of each filter
                                torch.mean(
                                    torch.cat(self.layer_outputs[layer]), dim=(1)
                                )
                                .squeeze()
                                .cpu()
                                .numpy()
                            )
                        else:
                            means = (  # Calculate the mean of the activations of each filter
                                torch.mean(
                                    torch.cat(self.layer_outputs[layer]), dim=(2, 3)
                                )
                                .squeeze()
                                .cpu()
                                .numpy()
                            )
                        for neuron, mean_of_neuron in enumerate(means):
                            if neuron not in layer_datas[layer]:
                                layer_datas[layer][neuron] = []
                            layer_datas[layer][neuron].append(mean_of_neuron)

            outputs[flicker_loader] = layer_datas
        self.remove_hooks()
        return outputs

    def calculate_snr(self, fft_values, bins=3, skip=1, scale_to_db=True):
        snr_values = []
        for i in range(len(fft_values)):
            # Define the noise baseline: left and right bins
            left_noise = fft_values[max(0, i - bins - skip):max(0, i - skip)]
            right_noise = fft_values[
                min(len(fft_values), i + skip + 1):min(len(fft_values), i + skip + bins + 1)
            ]
            noise_baseline = np.concatenate((left_noise, right_noise))

            # Avoid division by zero or negative values
            baseline_avg = np.mean(noise_baseline) if len(noise_baseline) > 0 else 1e-8
            baseline_avg = max(baseline_avg, 1e-8)  # Ensure positive baseline average
            fft_value = max(fft_values[i], 1e-8)  # Ensure positive FFT value

            # Calculate SNR
            snr = fft_value / baseline_avg
            if scale_to_db:
                snr = 10 * np.log10(snr)  # Convert to decibels
            if np.isinf(snr) or snr > 1e6:
                snr = 1e6  # Cap SNR value
                
            snr_values.append(max(snr, 0))  # Ensure non-negative SNR

        return np.array(snr_values)

    def calculate_snr_for_neuron(self, grads, bins=3, skip=1, scale_to_db=True):
        # Compute FFT magnitudes and remove DC component
        fft_output = np.abs(np.fft.rfft(grads) / len(grads))  # Normalize by input length
        fft_output = fft_output[1:]  # Remove DC component

        # Calculate SNR for each frequency component
        snr_values_for_neuron = self.calculate_snr(
            fft_output, bins=bins, skip=skip, scale_to_db=scale_to_db
        )

        # Return mean SNR value for interpretability
        mean = np.mean(snr_values_for_neuron)
        return mean


    def calculate_total_calc_snr(self):
        total = 0
        for loader in self.layer_datas:
            for layer in self.layer_datas[loader]:
                total += len(self.layer_datas[loader][layer])
        return total

    def get_snr_neurons(self):
        neurons = {}
        # with tqdm(
        #  total=self.calculate_total_calc_snr(), desc="Calculating SNR"
        # ) as pbar:
        for loader in self.layer_datas:
            for layer in self.layer_datas[loader]:
                if layer not in neurons:
                    neurons[layer] = {}
                for neuron in self.layer_datas[loader][layer]:
                    if neuron not in neurons[layer]:
                        neurons[layer][neuron] = []
                    snr_val = self.calculate_snr_for_neuron(
                        self.layer_datas[loader][layer][neuron]
                    )
                    neurons[layer][neuron].append(snr_val)
                    # pbar.update(1)
        for layer in neurons:
            for neuron in neurons[layer]:
                neurons[layer][neuron] = np.mean(
                    neurons[layer][neuron]
                )  # Mean SNRs across all flicker images
        return neurons

    def return_snr_neurons(self):
        return self.snr_neurons
