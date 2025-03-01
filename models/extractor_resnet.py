from typing import Dict, Iterable, Callable
from torch import nn

import torch
import torch.nn.functional as F

from models.resnet import ResNet
from models.simple import SimpleNet

simplenetactivations = ['conv1', 'conv2', 'fc1', 'fc2']




class FeatureExtractor:
    resnet18activations = ['layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0',
                           'layer4.1']

    def __init__(self, model):
        self.layer_names = list()
        layers = dict([*model.named_modules()]).keys()
        # print("****************************************")
        # print(layers)
        if isinstance(model, ResNet):
            for layer in layers:
                if 'relu' in layer or layer in self.resnet18activations:
                    self.layer_names.append(layer)
        elif isinstance(model, SimpleNet):
            for layer in layers:
                if layer in simplenetactivations:
                    self.layer_names.append(layer)
        else:
            raise NotImplemented

        self._extracted_activations = dict()
        self._extracted_grads = dict()
        self.hooks = list()

    def clear_activations(self):
        self._extracted_activations = dict()

    def save_activation(self, layer_name: str):
        def hook(module, input, output):
            if layer_name not in self._extracted_activations.keys():
                self._extracted_activations[layer_name] = output

        return hook

    def save_grads(self, layer_name: str):
        def hook(module, input, output):
            self._extracted_grads[layer_name] = output

        return hook

    def insert_activation_hook(self, model: nn.Module):
        named_modules = dict([*model.named_modules()])

        for name in self.layer_names:
            assert name in named_modules.keys()
            layer = named_modules[name]
            hook = layer.register_forward_hook(self.save_activation(name))
            self.hooks.append(hook)

    def insert_grads_hook(self, model: nn.Module):
        named_modules = dict([*model.named_modules()])
        for name in self.layer_names:
            assert name in named_modules.keys()
            layer = named_modules[name]
            layer.register_backward_hook(self.save_grads(name))

    def release_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def activations(self, model, module):
        if isinstance(model, ResNet):

            return self._extracted_activations[module]
        if isinstance(model, SimpleNet):
            return F.relu(self._extracted_activations[module])