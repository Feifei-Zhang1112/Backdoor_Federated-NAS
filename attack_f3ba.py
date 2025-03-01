import logging
from typing import Dict

from collections import OrderedDict
from collections import defaultdict
import torch
import torch.nn as nn
import copy
import numpy as np
from models.model import Model
from models.nc_model import NCModel
from torch.utils.data import Dataset, DataLoader
from losses.loss_functions import compute_all_losses_and_grads, trigger_attention_loss, trigger_loss
from utils.min_norm_solvers import MGDASolver
from utils.parameters import Params
from synthesizers.pattern_synthesizer import *
from models.extractor import FeatureExtractor 

from models.resnet import ResNet
from models.simple import SimpleNet
from tasks.fl.cifarfed_task import CifarFedTask
# from tasks.fl.fl_emnist_task import EMNISTFedTask
# from tasks.fl.timagefed_task import TinyImageNetFedTask
from darts.model import NetworkCIFAR
# from darts.model import NetworkEMNIST
# from darts.model import NetworkTinyImageNet
from tasks.batch import Batch

from darts.model_search import Network
#from models.resnet import layer2module
from darts.model import layer2module
import torch.nn.functional as F
import gc
import PIL.Image

logger = logging.getLogger('logger')


def ensure_tensors_on_current_device(*tensors):
    current_device_index = torch.cuda.current_device()
    current_device = torch.device(f"cuda:{current_device_index}")

    moved_tensors = []
    for tensor in tensors:
        if tensor.device != current_device:
            moved_tensors.append(tensor.to(current_device))
        else:
            moved_tensors.append(tensor)
    return tuple(moved_tensors)




def adjust_scaled_patterns4(conv_kernel, scaled_pattern):
    if conv_kernel.shape == scaled_pattern.shape:
        return scaled_pattern

    c0_diff = conv_kernel.shape[0] - scaled_pattern.shape[0]
    if c0_diff > 0:
        scaled_pattern = F.pad(scaled_pattern, (0, 0, 0, 0, 0, c0_diff), "constant", 0)
    elif c0_diff < 0:
        scaled_pattern = scaled_pattern[:conv_kernel.shape[0]]

    padding = [max(s1 - s2, 0) for s1, s2 in zip(conv_kernel.shape[1:], scaled_pattern.shape[1:])]
    cropping = [max(s2 - s1, 0) for s1, s2 in zip(conv_kernel.shape[1:], scaled_pattern.shape[1:])]

    if any(p > 0 for p in padding):
        scaled_pattern = F.pad(scaled_pattern, (padding[1], padding[1], padding[0], padding[0]), "constant", 0)

    if any(c > 0 for c in cropping):
        scaled_pattern = scaled_pattern[:, cropping[0]:(scaled_pattern.shape[1] - cropping[0]),
                         cropping[1]:(scaled_pattern.shape[2] - cropping[1])]

    if conv_kernel.shape[1:] != scaled_pattern.shape[1:]:
        if scaled_pattern.shape[1] == 0 or scaled_pattern.shape[2] == 0:
            scaled_pattern = torch.ones((scaled_pattern.shape[0], 1, 1), dtype=scaled_pattern.dtype,
                                        device=scaled_pattern.device)
        scaled_pattern = F.interpolate(scaled_pattern.unsqueeze(0), size=conv_kernel.shape[1:3], mode='bilinear',
                                       align_corners=True).squeeze(0)

    return scaled_pattern


def print_named_modules(model):
    for name, module in model.named_modules():
        print(f"Module name: {name}, Module type: {type(module)}")


def get_conv_weight_names(model: nn.Module):  # search
    conv_targets = []

    for name, module in model.stem.named_modules():
        if isinstance(module, nn.Conv2d):
            weight_name = 'stem.' + name + '.weight'
            conv_targets.append(weight_name)

    # if isinstance(model, (NetworkCIFAR, NetworkEMNIST, NetworkTinyImageNet)):
    if isinstance(model, (NetworkCIFAR)):
        cell_container = model.cells
        aux_module = model.auxiliary_head if hasattr(model, 'auxiliary_head') else None
    else:
        cell_container = model.cells
        aux_module = None

    # search
    # for i, cell in enumerate(model.cells):
    for i, cell in enumerate(cell_container):
        for name, module in cell.named_modules():
            if isinstance(module, nn.Conv2d):
                weight_name = f'cells.{i}.' + name + '.weight'
                conv_targets.append(weight_name)

    # search
    # if hasattr(model, 'auxiliary_head'):
    if aux_module:
        for name, module in aux_module.named_modules():
            # for name, module in model.auxiliary_head.named_modules():
            if isinstance(module, nn.Conv2d):
                weight_name = 'auxiliary_head.' + name + '.weight'
                conv_targets.append(weight_name)

    return conv_targets

def get_accuracy(model, task, loader):
    for metric in task.metrics:
        metric.reset_metric()

    model.eval()
    specified_metrics = ['AccuracyMetric']
    for i, data in enumerate(loader):
        batch = task.get_batch(i, data)
        outputs = model(batch.inputs)
        '''To Modify'''
        task.accumulate_metrics(outputs, batch.labels, specified_metrics=specified_metrics)

    accuracy = None
    for metric in task.metrics:
        if metric.__class__.__name__ in specified_metrics:
            accuracy = metric.get_value()

    return accuracy['Top-1']

#
def test_handcrafted_acc(model, target, id, task, loader):
    weights = model.state_dict()
    cur_conv_kernel = weights[target][id, ...].clone().detach()
    weights[target][id, ...] = 0
    accuracy = get_accuracy(model, task, loader)
    weights[target][id, ...] = cur_conv_kernel
    return accuracy

def get_neuron_weight_names(model: nn.Module):  # search
    neuron_targets = []

    # search

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        neuron_targets.append('classifier.weight')

    if hasattr(model, 'auxiliary_head'):
        for name, module in model.auxiliary_head.named_modules():
            if isinstance(module, nn.Linear):
                weight_name = 'auxiliary_head.' + name + '.weight'
                neuron_targets.append(weight_name)

    if hasattr(model, 'cells'):
        for cell_idx, cell in enumerate(model.cells):
            if hasattr(cell, 'classifier') and isinstance(cell.classifier, nn.Linear):
                weight_name = f'cells.{cell_idx}.classifier.weight'
                neuron_targets.append(weight_name)

    return neuron_targets


class Attack_F3BA:
    params: Params
    synthesizer: PatternSynthesizer
    nc_model: Model
    nc_optim: torch.optim.Optimizer
    loss_hist = list()
    # fixed_model: Model

    # weight_modification attack
    nc_p_norm: int
    acc_threshold: int

    def __init__(self, params, synthesizer, args):
        self.params = params
        self.synthesizer = synthesizer
        self.args = args
        # F3BA
        self.loss_tasks = self.params.loss_tasks.copy()
        self.loss_balance = self.params.loss_balance
        self.mgda_normalize = self.params.mgda_normalize
        self.backdoor_label = params.backdoor_label
        self.handcraft = params.handcraft
        self.acc_threshold = params.acc_threshold if params.handcraft else 0
        self.handcraft_trigger = params.handcraft_trigger
        self.kernel_selection = params.kernel_selection
        self.raw_model = None
        self.neurotoxin = True if params.backdoor == 'neurotoxin' else False
        self.w_reduce = None
        self.w_normal = None

        self.means = (0.4914, 0.4822, 0.4465)
        self.lvars = (0.2023, 0.1994, 0.2010)
        # NC hyper params
        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc_model = NCModel(params.input_shape[1]).to(params.device)
            self.nc_optim = torch.optim.Adam(self.nc_model.parameters(), 0.01)
        if 'mask_norm' in self.params.loss_tasks:
            self.nc_p_norm = self.params.nc_p_norm
        if self.kernel_selection == "movement":
            self.previous_global_model = None



    def compute_blind_loss(self, model, criterion, batch, attack, train=False):
        """

        :param model:
        :param criterion:
        :param batch:
        :param attack: Do not attack at all. Ignore all the parameters
        :return:
        """
        batch = batch.clip(self.params.clip_batch)
        loss_tasks = self.params.loss_tasks.copy() if attack else ['normal']
        batch_back = self.synthesizer.make_backdoor_batch(batch=batch, test=False, attack=attack)
        scale = dict()

        if 'neural_cleanse' in loss_tasks:
            self.neural_cleanse_part1(model, batch, batch_back)

        if self.params.loss_threshold and (np.mean(self.loss_hist) >= self.params.loss_threshold
                                           or len(self.loss_hist) < 1000):
            loss_tasks = ['normal']

        if len(loss_tasks) == 1:
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, train=train, compute_grad=False
            )
        elif self.params.loss_balance == 'MGDA':

            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, train=train, compute_grad=True)
            if len(loss_tasks) > 1:
                scale = MGDASolver.get_scales(grads, loss_values,
                                              self.params.mgda_normalize,
                                              loss_tasks)
        elif self.params.loss_balance == 'fixed':
            loss_values, grads = compute_all_losses_and_grads(
                loss_tasks,
                self, model, criterion, batch, batch_back, train=train, compute_grad=False)

            for t in loss_tasks:
                scale[t] = self.params.fixed_scales[t]
        else:
            raise ValueError(f'Please choose between `MGDA` and `fixed`.')

        if len(loss_tasks) == 1:
            scale = {loss_tasks[0]: 1.0}
        self.loss_hist.append(loss_values['normal'].item())
        self.loss_hist = self.loss_hist[-1000:]
        blind_loss = self.scale_losses(loss_tasks, loss_values, scale)

        return blind_loss

    def search_candidate_weights(self, model: Model, proportion=0.2):
        assert self.kernel_selection in ['random', 'movement']
        candidate_weights = OrderedDict()
        model_weights = model.state_dict()

        n_labels = 0

        if self.kernel_selection == "movement":
            history_weights = self.previous_global_model.state_dict()
            for layer in history_weights.keys():
                if 'conv' in layer:
                    proportion = self.params.conv_rate
                elif 'fc' in layer:
                    proportion = self.params.fc_rate

                candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                n_weight = candidate_weights[layer].numel()
                theta = torch.sort(candidate_weights[layer].flatten(), descending=False)[0][int(n_weight * proportion)]
                candidate_weights[layer][candidate_weights[layer] < theta] = 1
                candidate_weights[layer][candidate_weights[layer] != 1] = 0

        return candidate_weights

    def flip_filter_as_trigger(self, conv_kernel: torch.Tensor, difference):
        flip_factor = self.params.flip_factor
        c_min, c_max = conv_kernel.min(), conv_kernel.max()
        pattern = None
        if difference is None:
            pattern_layers, _ = self.synthesizer.get_pattern()
            x_top, y_top = self.synthesizer.x_top, self.synthesizer.y_top
            x_bot, y_bot = self.synthesizer.x_bot, self.synthesizer.y_bot
            pattern = pattern_layers[:, x_top:x_bot, y_top:y_bot]
        else:
            pattern = difference
        w = conv_kernel[0, ...].size()[0]
        resize = transforms.Resize((w, w), interpolation=PIL.Image.NEAREST)

        scaled_patterns = [F.interpolate(p.unsqueeze(0).unsqueeze(0), size=(w, w)) for p in pattern]
        scaled_pattern = torch.cat(scaled_patterns, dim=0).squeeze()

        p_min, p_max = pattern.min(), pattern.max()

        # tensor movement
        # if self.params.fl_number_of_adversaries > 1:
        current_device_index = torch.cuda.current_device()
        pattern, p_min, p_max, c_max, c_min = \
            ensure_tensors_on_current_device(pattern, p_min, p_max, c_max, c_min)
        scaled_pattern = (pattern - p_min) / (p_max - p_min) * (c_max - c_min) + c_min
        # print("Before cropping:", scaled_pattern.shape)
        scaled_pattern, conv_kernel = ensure_tensors_on_current_device(scaled_pattern, conv_kernel)
        # if the size of scaled_pattern is no small than conv (>=)
        scaled_pattern = adjust_scaled_patterns4(conv_kernel, scaled_pattern)

        # if the size of the scaled is smaller than conv
        # adjust_kernel(conv_kernel, scaled_pattern, flip_factor)

        return conv_kernel

    def calculate_activation_difference(self, raw_model, new_model, layer_name, kernel_ids, task, loader: DataLoader):
        raw_extractor, new_extractor = FeatureExtractor(raw_model), FeatureExtractor(new_model)
        raw_extractor.insert_activation_hook(raw_model)
        new_extractor.insert_activation_hook(new_model)
        difference = None
        for i, data in enumerate(loader):
            batch = task.get_batch(i, data)
            batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
            raw_outputs = raw_model(batch.inputs)
            new_outputs = new_model(batch.inputs)
            module = layer2module(new_model, layer_name, self.args)
            # modify this
            raw_batch_activations = raw_extractor.activations(raw_model, module)[:, kernel_ids, ...]
            new_batch_activations = new_extractor.activations(new_model, module)[:, kernel_ids, ...]
            batch_activation_difference = new_batch_activations - raw_batch_activations
            # mean_difference = torch.mean(batch_activation_difference, [0, 1])
            mean_difference = torch.mean(batch_activation_difference, [0])
            difference = difference + mean_difference if difference is not None else mean_difference

        difference = difference / len(loader)

        raw_extractor.release_hooks()
        new_extractor.release_hooks()

        return difference

    # possible overlap
    def conv_features(self, model, task, loader, attack):
        features = None

        with torch.no_grad():
            for i, data in enumerate(loader):
                batch = task.get_batch(i, data)
                batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
                feature = model.features(batch.inputs).mean([0])
                features = feature if features is None else features + feature
                del feature
            avg_features = features / len(loader)

        return avg_features

    def calculate_feature_difference(self, raw_model, new_model, task, loader):
        diffs = None
        for i, data in enumerate(loader):
            batch = task.get_batch(i, data)
            batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
            diff = new_model.features(batch.inputs).mean([0]) - raw_model.features(batch.inputs).mean([0])
            diffs = diff if diffs is None else diffs + diff

        avg_diff = diffs / len(self.train_local)
        return avg_diff

    def conv_activation(self, model, layer_name, task, loader, attack):
        extractor = FeatureExtractor(model)
        extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name, self.args)

        if self.args.stage == "search":
            total_samples = 0
            accum_activation = 0
            with torch.no_grad():  # Disable gradient computation
                for i, data in enumerate(loader):
                    batch = task.get_batch(i, data)
                    batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)

                    _ = model(batch.inputs)
                    conv_activation = extractor.activations(module)

                    # Online computation of average
                    batch_size = conv_activation.size(0)
                    total_samples += batch_size
                    accum_activation += torch.sum(conv_activation, dim=0)

                    # Release variables and clear GPU memory
                    del batch
                    del conv_activation
                    torch.cuda.empty_cache()
            avg_activation = accum_activation / total_samples

        else:
            conv_activations = None
            for i, data in enumerate(loader):
                batch = task.get_batch(i, data)
                batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
                _ = model(batch.inputs)
                conv_activation = extractor.activations(module)
                # conv_activation = extractor.activations(model, module)
                conv_activation = torch.mean(conv_activation, [0])
                conv_activations = conv_activation if conv_activations is None else conv_activations + conv_activation
            avg_activation = conv_activations / len(loader)

        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation

    def fc_activation(self, model: Model, layer_name, task, loader, attack):
        extractor = FeatureExtractor(model)
        extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name, self.args)

        if self.args.stage == "search":
            total_samples = 0
            accum_activation = 0
            with torch.no_grad():
                for i, data in enumerate(loader):
                    batch = task.get_batch(i, data)


                    batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)

                    _ = model(batch.inputs)
                    neuron_activation = extractor.activations(module)

                    batch_size = neuron_activation.size(0)
                    total_samples += batch_size
                    accum_activation += torch.sum(neuron_activation, dim=0)

                    del batch
                    del neuron_activation
                    torch.cuda.empty_cache()

            avg_activation = accum_activation / total_samples


        else:
            neuron_activations = None
            for i, data in enumerate(loader):
                batch = task.get_batch(i, data)


                batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
                _ = model(batch.inputs)
                neuron_activation = extractor.activations(module)
                # neuron_activation = extractor.activations(model, module)
                neuron_activations = neuron_activation if neuron_activations is None else neuron_activations + neuron_activation

            avg_activation = neuron_activations / len(self.train_local)

        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation

    def inject_handcrafted_filters(self, model, candidate_weights, task, loader):
        conv_weight_names = get_conv_weight_names(model)
        difference = None
        for layer_name, conv_weights in candidate_weights.items():
            if layer_name not in conv_weight_names:
                continue
            model_weights = model.state_dict()
            n_filter = conv_weights.size()[0]
            for i in range(n_filter):
                conv_kernel = model_weights[layer_name][i, ...].clone().detach()
                handcrafted_conv_kernel = self.flip_filter_as_trigger(conv_kernel, difference)
                # handcrafted_conv_kernel = conv_kernel

                mask = conv_weights[i, ...]

                # if self.params.fl_number_of_adversaries > 1:

                # tensor movement
                mask, handcrafted_conv_kernel, model_weights[layer_name] = \
                    ensure_tensors_on_current_device(mask, handcrafted_conv_kernel, model_weights[layer_name])

                model_weights[layer_name][i, ...] = mask * handcrafted_conv_kernel + (1 - mask) * \
                                                    model_weights[layer_name][i, ...]
                # model_weights[layer_name][i, ...].mul_(1-mask)
                # model_weights[layer_name][i, ...].add_(mask * handcrafted_conv_kernel)

            model.load_state_dict(model_weights)
            difference = self.conv_activation(model, layer_name, task, loader, True) - self.conv_activation(model,
                                                                                                            layer_name,
                                                                                                            task,
                                                                                                            loader,
                                                                                                            False)
            print("handcraft_conv: {}".format(layer_name))

        torch.cuda.empty_cache()
        if difference is not None:
            difference = None
            gc.collect()

            feature_difference = self.conv_features(model, task, loader, True) \
                                 - self.conv_features(model, task, loader, False)
            return feature_difference

    def set_handcrafted_filters2(self, model: Model, candidate_weights, layer_name):
        conv_weights = candidate_weights[layer_name]
        # print("check candidate:",int(torch.sum(conv_weights)))
        model_weights = model.state_dict()
        temp_weights = copy.deepcopy(model_weights[layer_name])

        n_filter = conv_weights.size()[0]

        for i in range(n_filter):
            conv_kernel = model_weights[layer_name][i, ...].clone().detach()
            handcrafted_conv_kernel = self.flip_filter_as_trigger(conv_kernel, difference=None)
            mask = conv_weights[i, ...]

            # if self.params.fl_number_of_adversaries > 1:
            # tensor movement
            mask, handcrafted_conv_kernel, conv_kernel, model_weights[layer_name] = ensure_tensors_on_current_device(
                mask, handcrafted_conv_kernel, conv_kernel, model_weights[layer_name])

            model_weights[layer_name][i, ...] = mask * handcrafted_conv_kernel + (1 - mask) * model_weights[layer_name][
                i, ...]

        model.load_state_dict(model_weights)
        # n_turn=int(torch.sum(torch.sign(temp_weights)!=torch.sign(model_weights[layer_name])))
        # print("check modify:",n_turn)

    def optimize_backdoor_trigger(self, model: Model, candidate_weights, task, loader):
        pattern, mask = self.synthesizer.get_pattern()
        pattern.requires_grad = True

        x_top, y_top = self.synthesizer.x_top, self.synthesizer.y_top
        x_bot, y_bot = self.synthesizer.x_bot, self.synthesizer.y_bot

        cbots, ctops = list(), list()
        for h in range(pattern.size()[0]):
            cbot = (0 - self.means[h]) / self.lvars[h]
            ctop = (1 - self.means[h]) / self.lvars[h]
            cbots.append(round(cbot, 2))
            ctops.append(round(ctop, 2))

        raw_weights = copy.deepcopy(model.state_dict())

        self.set_handcrafted_filters2(model, candidate_weights, "stem.0.weight")
        for epoch in range(2):
            losses = list()
            for i, data in enumerate(loader):
                batch_size = self.params.batch_size
                clean_batch, backdoor_batch = task.get_batch(i, data), task.get_batch(i, data)
                mask, pattern = ensure_tensors_on_current_device(mask, pattern)
                backdoor_batch.inputs[:batch_size] = (1 - mask) * backdoor_batch.inputs[:batch_size] + mask * pattern
                backdoor_batch.labels[:batch_size].fill_(self.params.backdoor_label)

                self.set_handcrafted_filters2(model, candidate_weights, "stem.0.weight")

                loss, grads = trigger_loss(model, backdoor_batch.inputs, clean_batch.inputs, pattern, self.params,
                                           grads=True)
                losses.append(loss.item())

                pattern = pattern + grads[0] * 0.1

                n_channel = pattern.size()[0]
                for h in range(n_channel):
                    pattern[h, x_top:x_bot, y_top:y_bot] = torch.clamp(pattern[h, x_top:x_bot, y_top:y_bot], cbots[h],
                                                                       ctops[h], out=None)

                model.zero_grad()
            print("epoch:{} trigger loss:{}".format(epoch, np.mean(losses)))

        print(pattern[0, x_top:x_bot, y_top:y_bot].cpu().data)

        self.synthesizer.pattern = pattern.clone().detach()
        self.synthesizer.pattern_tensor = pattern[0, x_top:x_bot, y_top:y_bot].clone().detach()

        model.load_state_dict(raw_weights)
        torch.cuda.empty_cache()

        return self.synthesizer.pattern_tensor

    def inject_handcrafted_neurons(self, model, candidate_weights, task, diff, loader):
        handcrafted_connectvites = defaultdict(list)
        target_label = self.params.backdoor_label

        n_labels = -1
        if self.params.task == 'CIFAR':
            n_labels = 10
        elif self.params.task == 'EMNIST':
            n_labels = 47
        elif self.params.task == 'TINY':
            n_labels = 80
        print("n_labels:", n_labels)
        fc_names = get_neuron_weight_names(model)
        fc_diff = diff
        last_layer, last_ids = None, list()
        for layer_name, connectives in candidate_weights.items():
            if layer_name not in fc_names:
                continue
            raw_model = copy.deepcopy(model)
            model_weights = model.state_dict()
            ideal_signs = torch.sign(fc_diff)
            n_next_neurons = connectives.size()[0]
            # last_layer
            if n_next_neurons == n_labels:
                break

            # if self.params.fl_number_of_adversaries > 1:
            # tensor movement
            fc_diff, ideal_signs, model_weights[layer_name], connectives = \
                ensure_tensors_on_current_device(fc_diff, ideal_signs, model_weights[layer_name], connectives)

            ideal_signs = ideal_signs.repeat(n_next_neurons, 1) * connectives
            # count the flip num
            n_flip = torch.sum(((ideal_signs * torch.sign(model_weights[layer_name]) * connectives == -1).int()))
            print("n_flip in {}:{}".format(layer_name, n_flip))
            model_weights[layer_name] = (1 - connectives) * model_weights[layer_name] + torch.abs(
                connectives * model_weights[layer_name]) * ideal_signs
            model.load_state_dict(model_weights)
            last_layer = layer_name
            fc_diff = self.fc_activation(model, layer_name, task, loader, attack=True).mean([0]) - self.fc_activation(
                model, layer_name, task, loader, attack=False).mean([0])

    # ======original=====
    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for it, t in enumerate(loss_tasks):
            self.params.running_losses[t].append(loss_values[t].item())
            self.params.running_scales[t].append(scale[t])
            if it == 0:
                blind_loss = scale[t] * loss_values[t]
            else:
                blind_loss += scale[t] * loss_values[t]
        self.params.running_losses['total'].append(blind_loss.item())
        return blind_loss

    def neural_cleanse_part1(self, model, batch, batch_back):
        self.nc_model.zero_grad()
        model.zero_grad()

        self.nc_model.switch_grads(True)
        model.switch_grads(False)
        output = model(self.nc_model(batch.inputs))
        nc_tasks = ['neural_cleanse_part1', 'mask_norm']

        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        loss_values, grads = compute_all_losses_and_grads(nc_tasks,
                                                          self, model,
                                                          criterion, batch,
                                                          batch_back,
                                                          compute_grad=False
                                                          )
        # Using NC paper params
        logger.info(loss_values)
        loss = 0.999 * loss_values['neural_cleanse_part1'] + 0.001 * loss_values['mask_norm']
        loss.backward()
        self.nc_optim.step()

        self.nc_model.switch_grads(False)
        model.switch_grads(True)

    def fl_scale_update(self, local_update: Dict[str, torch.Tensor]):
        for name, value in local_update.items():
            value.mul_(self.params.fl_weight_scale)
