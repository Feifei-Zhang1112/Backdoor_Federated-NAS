import logging
from typing import Dict
import copy
import torch
from copy import deepcopy
import numpy as np
from models.model import Model
from models.nc_model import NCModel
from losses.loss_functions import compute_all_losses_and_grads
from utils.min_norm_solvers import MGDASolver
from utils.parameters import Params
from synthesizers.pattern_synthesizer import *
import time

logger = logging.getLogger('logger')


class Attack_A3FL:
    params: Params
    synthesizer: PatternSynthesizer
    nc_model: Model
    nc_optim: torch.optim.Optimizer
    loss_hist = list()

    def __init__(self, params, synthesizer, args):
        self.params = params
        self.synthesizer = synthesizer
        self.args = args

        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc_model = NCModel(params.input_shape[1]).to(params.device)
            self.nc_optim = torch.optim.Adam(self.nc_model.parameters(), 0.005)

    def get_adv_model(self, model, train_queue, trigger, mask, device):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr=self.args.learning_rate,
                                  momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        print("Optimize the trigger.")

        for step, (input, target) in enumerate(train_queue):
            print("Optimization step {:04d}".format(step))
            input = input.to(device)
            target = target.to(device)
            mask = mask.to(device)
            trigger = trigger.to(device)
            input = trigger * mask + (1 - mask) * input
            outputs = adv_model(input)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = ce_loss(outputs, target)
            adv_opt.zero_grad()
            loss.backward()
            adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name or 'op' in name or 'preprocessing' in name or '_ops' in name:
                if (dict(adv_model.named_parameters())[name].grad is not None and
                        dict(model.named_parameters())[name].grad is not None):
                    sim_count += 1
                    sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),
                                        dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum / sim_count if sim_count > 0 else 0.0

    def search_trigger(self, model, train_queue, device, iter, adversary_id=0, epoch=0):
        model.eval()
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = 0.01

        normal_grad = 0.
        count = 0
        t = self.synthesizer.pattern.clone().detach()
        m = self.synthesizer.mask.clone().detach()
        t.requires_grad = True
        adv_model, adv_w = self.get_adv_model(model, train_queue, t, m, device)
        logging.info("start A3FL trigger fine-tuning")

        print("Optimize the trigger.")
        adv_model.train()
        t.requires_grad = True

        for step, (input, target) in enumerate(train_queue):
            count += 1
            print("Optimization count: ", count)

            t.requires_grad = True
            input, target = input.to(device), target.to(device)

            batch = Batch(step, input, target)
            batch = batch.to(device)
            batch_back = self.synthesizer.make_backdoor_batch(batch=batch, test=False, attack=True)
            target = batch_back.labels.to(device)

            outputs = adv_model(input)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            nm_loss = ce_loss(outputs, target)
            loss = 0.01 * adv_w * nm_loss

            loss.backward()

            if t.grad is None:
                print("grads is None at step:", step)
            else:
                print("grads is not None at step:", step)
                print(f"grads: {t.grad}")
            if t.grad is not None:
                normal_grad += t.grad.sum()
                with torch.no_grad():
                    t -= alpha * t.grad.sign()
                    t = torch.clamp(t, min=-2, max=2)
                    self.synthesizer.pattern = t

            t.requires_grad = True

        self.synthesizer.pattern = t
        self.synthesizer.mask = m
        x_top, y_top = self.synthesizer.x_top, self.synthesizer.y_top
        x_bot, y_bot = self.synthesizer.x_bot, self.synthesizer.y_bot
        self.synthesizer.pattern_tensor = t[0, x_top:x_bot, y_top:y_bot].clone().detach()
        t = t.detach()

    def compute_blind_loss(self, model, criterion, batch, attack, train=False):
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
        logger.info(loss_values)
        loss = 0.999 * loss_values['neural_cleanse_part1'] + 0.001 * loss_values['mask_norm']
        loss.backward()
        self.nc_optim.step()

        self.nc_model.switch_grads(False)
        model.switch_grads(True)

    def fl_scale_update(self, local_update: Dict[str, torch.Tensor]):
        for name, value in local_update.items():
            value.mul_(self.params.fl_weight_scale)
