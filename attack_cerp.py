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
from torch.autograd import Variable
from models.resnet import ResNet
from models.simple import SimpleNet
from tasks.fl.cifarfed_task import CifarFedTask
from darts.model import NetworkCIFAR
from tasks.batch import Batch
from darts.model_search import Network
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


def cuda(tensor, is_cuda):
    if is_cuda:
        return tensor.cuda()
    else:
        return tensor


def proj_lp(v, xi, p):
    if p == 2:
        v = v * min(1, xi / torch.norm(v))
    elif p == np.inf:
        v = torch.clamp(v, -xi, xi)
    else:
        raise ValueError('Values of p different from 2 and Inf are currently not supported...')
    return v


def get_accuracy(model, task, loader):
    for metric in task.metrics:
        metric.reset_metric()

    model.eval()
    specified_metrics = ['AccuracyMetric']
    for i, data in enumerate(loader):
        batch = task.get_batch(i, data)
        outputs = model(batch.inputs)
        task.accumulate_metrics(outputs, batch.labels, specified_metrics=specified_metrics)

    accuracy = None
    for metric in task.metrics:
        if metric.__class__.__name__ in specified_metrics:
            accuracy = metric.get_value()

    return accuracy['Top-1']


class Attack_CERP:
    def __init__(self, params, synthesizer, args):
        self.params = params
        self.synthesizer = synthesizer
        self.args = args
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

        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc_model = NCModel(params.input_shape[1]).to(params.device)
            self.nc_optim = torch.optim.Adam(self.nc_model.parameters(), 0.01)
        if 'mask_norm' in self.params.loss_tasks:
            self.nc_p_norm = self.params.nc_p_norm
        if self.kernel_selection == "movement":
            self.previous_global_model = None

    def cerp_trigger(self, local_model, target_model, noise_trigger, initial_trigger, train_queue, device):
        logging.info("start CERP trigger fine-tuning")
        model = copy.deepcopy(local_model)
        model.load_state_dict(target_model.state_dict())
        model.eval()
        pre_trigger = torch.tensor(noise_trigger).cuda()
        aa = copy.deepcopy(initial_trigger).cuda()

        noise = copy.deepcopy(pre_trigger)
        for step, (input, target) in enumerate(train_queue):
            noise = Variable(cuda(noise, True), requires_grad=True)
            input = input.to(device)
            target = torch.Tensor(target.numpy()).long()
            target = target.to(device)
            batch = Batch(step, input, target)
            batch = batch.to(device)
            batch_back = self.synthesizer.make_backdoor_batch(batch=batch, test=False, attack=True)
            inputs = batch_back.inputs.to(device)
            target = batch_back.labels.to(device)

            output = model(inputs)
            if isinstance(output, tuple):
                output = output[0]
            class_loss = nn.functional.cross_entropy(output, target)
            loss = class_loss
            model.zero_grad()
            if noise.grad:
                noise.grad.fill_(0)
            loss.backward(retain_graph=True)
            if noise.grad:
                noise = noise - noise.grad * 0.1

            delta_noise = noise - aa
            noise = aa + proj_lp(delta_noise, 10, 2)
            self.synthesizer.pattern = noise

            noise = Variable(cuda(noise.data, True), requires_grad=True)

        self.synthesizer.pattern = noise.clone().detach()
        x_top, y_top = self.synthesizer.x_top, self.synthesizer.y_top
        x_bot, y_bot = self.synthesizer.x_bot, self.synthesizer.y_bot
        self.synthesizer.pattern_tensor = noise[0, x_top:x_bot, y_top:y_bot].clone().detach()

        return noise

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
        # self.loss_hist.append(loss_values['normal'].item())
        # self.loss_hist = self.loss_hist[-1000:]
        blind_loss = self.scale_losses(loss_tasks, loss_values, scale)

        return blind_loss

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
