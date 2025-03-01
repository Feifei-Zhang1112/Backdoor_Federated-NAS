from typing import Dict, Iterable, Callable
from torch import nn

import torch
import torch.nn.functional as F

from models.resnet import ResNet
from models.simple import SimpleNet


# F3BA
class FeatureExtractor:
    def __init__(self, model):
        self._extracted_activations = dict()
        self._extracted_grads = dict()
        self.hooks = list()

    def clear_activations(self):
        self._extracted_activations = dict()

    def save_activation(self, layer_name: str) -> Callable:
        def hook(module, input, output):
            if layer_name not in self._extracted_activations.keys():
                self._extracted_activations[layer_name] = output
        return hook

    def save_grads(self, layer_name: str) -> Callable:
        def hook(module, input, output):
            self._extracted_grads[layer_name] = output
        return hook

    def insert_activation_hook(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.ReLU, nn.BatchNorm2d)):
                hook = module.register_forward_hook(self.save_activation(name))
                self.hooks.append(hook)
            elif isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.ReLU, nn.BatchNorm2d)):  # Handle Sequential containers
                for sub_name, sub_module in module.named_modules():
                    full_name = name + "." + sub_name
                    if isinstance(sub_module, nn.Conv2d):
                        hook = sub_module.register_forward_hook(self.save_activation(full_name))
                        self.hooks.append(hook)

    def insert_grads_hook(self, model: nn.Module):
        named_modules = dict([*model.named_modules()])
        for name, module in named_modules.items():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.ReLU, nn.BatchNorm2d)):
                module.register_backward_hook(self.save_grads(name))

    def release_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def activations(self, module_name):
        return self._extracted_activations[module_name]

