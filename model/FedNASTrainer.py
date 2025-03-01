import logging

import numpy as np
import torch
from torch import nn
from torchsummaryX import summary
import copy
from darts import utils, genotypes, genotype_attention, tiny_geno_genotypes, cifar_geno_genotype
from darts.architect import Architect
from darts.model_genospace_cifar import NetworkCIFAR_GENO_CIFAR
from darts.model import NetworkCIFAR
from darts.model_attention import NetworkATTENTION
from darts.model_search_genospace_cifar import Network_GENO_CIFAR
# from darts.model_search import Network
# from darts.tiny_geno_model_search import Network
from darts.path_search import *
from utils.parameters import Params
from synthesizers.pattern_synthesizer import *
from tasks.batch import Batch
from torch.utils.data import DataLoader, Subset
from tasks.fl.cifarfed_task import CifarFedTask
from torch.autograd import Variable
from tasks.fl.timagefed_task import TinyImageNetFedTask
from darts.model_tiny_geno import NetworkGENOTINY
from models.resnet import resnet18
from losses.loss_functions import model_similarity_loss, model_dist_norm_var
import numpy as np
from data_preprocessing.data_loader import get_dataloader



class FedNASTrainer(object):

    def __init__(self, client_index, train_local, test_local, train_global, local_sample_number,
                 all_train_data_num, device, args, params, attack, attacker):
        self.client_index = client_index
        self.train_local = train_local  # equal to the train_loader in F3BA
        self.test_local = test_local
        self.local_sample_number = local_sample_number
        self.all_train_data_num = all_train_data_num
        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.adv_criterion = nn.CrossEntropyLoss().to(self.device)
        self.model = self.init_model()
        self.global_model = None
        self.model.to(self.device)
        self.attack = attack
        self.train_global = train_global


        self.attacker = attacker
        self.syn = attacker.synthesizer
        self.round = 0
        self.handcraft_rnd = 0  # F3BA
        self.train_n_samples = len(train_local.dataset)  # F3BA
        self.test_n_samples = len(test_local.dataset)  # F3BA
        self.params = params
        self.genotype = None  # best genotype
        self.gen_manager = None
        self.batch_idxs = None
        self.pattern = None  # for make_backdoor_batch
        self.previous_model = None  # cerp
        self.current_model = None  # cerp
        self.normal_params_variables = dict()  # cerp


        if params.task == 'CIFAR':
            self.task = CifarFedTask(params=self.params, loader=train_local)
            print("Cifar10.")
        elif params.task == 'TINY':
            self.task = TinyImageNetFedTask(params=self.params, loader=train_local)
            print("TinyImageNet.")

        # Search
        if self.attack and self.args.stage == "search" and self.args.attack_type == 'f3ba':
            self.gen_manager = GenotypeManager(self.model)
            self.genotype_for_exp = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1),
            ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5],reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

            self.gen_manager.set_genotype(self.genotype_for_exp)
        
        # F3BA
        if self.attack and self.args.attack_type == 'f3ba':
            train_idxs = range(self.train_n_samples)
            test_idxs = range(self.test_n_samples)
            train_nt = int(0.8*self.train_n_samples)
            test_nt = int(0.8*self.test_n_samples)
            train_handcraft_dataset = Subset(train_local.dataset, train_idxs)
            test_handcraft_dataset = Subset(test_local.dataset, test_idxs)

            self.train_handcraft_loader = DataLoader(train_handcraft_dataset, batch_size=params.batch_size, shuffle=True,
                                                     num_workers=1, drop_last=True)
            self.test_handcraft_loader = DataLoader(test_handcraft_dataset, batch_size=params.test_batch_size, shuffle=True,
                                                     num_workers=1, drop_last=True)

    def init_model(self):
        if self.args.search_space == 'TINYGENO':
            from darts.tiny_geno_model_search import Network
        else:
            from darts.model_search import Network
        num_classes = 10
        if self.args.dataset == 'cifar10':
            num_classes = 10
        elif self.args.dataset == 'tiny':
            num_classes = 80
        if self.args.stage == "search":
            if self.args.search_space == "DARTS" or self.args.search_space == "TINYGENO":
                model = Network(self.args.init_channels, num_classes, self.args.layers, self.criterion, self.device)
            elif self.args.search_space == "GENOSPACE":
                model = Network_GENO_CIFAR(self.args.init_channels, num_classes, self.args.layers, self.criterion, self.device)
        elif self.args.stage == "resnet":
            model = resnet18(num_classes=num_classes)
        else:
            if self.args.model_arch == 'NetV2':
                genotype = genotype_attention.DARTS
                logging.info(genotype)
                model = NetworkATTENTION(self.args.init_channels, num_classes, self.args.layers, self.args.auxiliary, genotype)
            elif self.args.model_arch == 'NetV1':
                genotype = genotypes.DARTS
                logging.info(genotype)
                model = NetworkCIFAR(self.args.init_channels, num_classes, self.args.layers, self.args.auxiliary, genotype)
            elif self.args.model_arch == 'TINYSEARCH':
                genotype = tiny_geno_genotypes.DARTS
                logging.info(genotype)
                model = NetworkGENOTINY(self.args.init_channels, num_classes, self.args.layers, self.args.auxiliary,
                                     genotype)
            elif self.args.model_arch == 'CIFARSEARCH':
                genotype = cifar_geno_genotype.DARTS
                logging.info(genotype)
                model = NetworkCIFAR_GENO_CIFAR(self.args.init_channels, num_classes, self.args.layers, self.args.auxiliary,
                                     genotype)
        return model

    def update_syn(self, syn):
        self.syn = syn
        self.attacker.synthesizer = syn
        self.attacker.synthesizer.set_imal(self.client_index)
        return

    def get_scaled_weight(self, local_model, local_update):
        for name, value in local_update.items():
            model_weight = local_model.state_dict()[name]
            model_weight.add_(local_update[name])

    def get_fl_update(self, local_model, global_model):
        local_update = dict()
        for name, data in local_model.state_dict().items():
            local_update[name] = (data - global_model.state_dict()[name])
        return local_update

    def update_model(self, weights):
        logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)
        self.global_model = copy.deepcopy(self.model)
        self.global_model = self.global_model.to(self.device)

    def update_arch(self, alphas):
        logging.info("update_arch. client_index = %d" % self.client_index)
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    def get_batch(self, batch_id, data) -> Batch:
        inputs, labels = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(Params.device)

    # FEDRAD
    def distillation_logit(self):
        self.model.to(self.device)
        self.model.eval()
        logit = []
        for step, (input, target) in enumerate(self.train_global):
            input = input.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                output = self.model(input)
                if isinstance(output, tuple):
                    logit_i = output[0]
                else:
                    logit_i = output
                # logit_i, _ = self.model(input)
                logit.append(logit_i.cpu().detach())
            del input, target, logit_i
            torch.cuda.empty_cache()
        return logit

    # ESFL
    def sample_bernoulli(self, epsilon):
        epsilon_tensor = torch.tensor(epsilon, device=self.device)
        prob_of_1 = (torch.exp(epsilon_tensor) - 1) / (2 * torch.exp(epsilon_tensor))
        return torch.bernoulli(prob_of_1)

    # def local_dp(self, epsilon=2.0):
    #     for name, module in self.model.state_dict().items():
    #         weight = module.weight.data
    #         min_weight = torch.min(weight)
    #         max_weight = torch.max(weight)
    #         c_l = (min_weight + max_weight) / 2
    #         r_l = max_weight - c_l
    #         epsilon_tensor = torch.tensor(epsilon, device=self.device)
    #         for i in range(weight.numel()):
    #             mu = weight.view(-1)[i] - c_l
    #             b = self.sample_bernoulli(epsilon).item()
    #             if b == 1:
    #                 weight.view(-1)[i] = c_l + mu * (torch.exp(epsilon_tensor) + 1) / (torch.exp(epsilon_tensor) - 1)
    #             else:
    #                 weight.view(-1)[i] = c_l + mu * (torch.exp(epsilon_tensor) - 1) / (torch.exp(epsilon_tensor) + 1)
    #         module.weight.data = weight



    # ESFL
    def local_dp(self, epsilon=4.0):
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.data
                print(f'Module Name: {name}')
                min_weight = torch.min(weight)
                max_weight = torch.max(weight)
                c_l = (min_weight + max_weight) / 2
                r_l = max_weight - c_l
                epsilon_tensor = torch.tensor(epsilon, device=self.device)
                for i in range(weight.numel()):
                    mu = weight.view(-1)[i] - c_l
                    b = self.sample_bernoulli(epsilon).item()
                    if b == 1:
                        weight.view(-1)[i] = c_l + mu * (torch.exp(epsilon_tensor) + 1) / (torch.exp(epsilon_tensor) - 1)
                    else:
                        weight.view(-1)[i] = c_l + mu * (torch.exp(epsilon_tensor) - 1) / (torch.exp(epsilon_tensor) + 1)
                module.weight.data = weight

    def handcraft_search(self, task):
            self.handcraft_rnd = self.handcraft_rnd + 1

            print("the attacker is {}".format(self.client_index))
            model = self.gen_manager.modify_network_with_genotype(self.model)

            w_normal, w_reduce = self.gen_manager.get_weights_from_current_architecture(self.model)

            self.attacker.w_normal = w_normal
            self.attacker.w_reduce = w_reduce
            # model = model.cpu()
            model.eval()
            train_handcraft_loader, train_loader = self.train_handcraft_loader, self.train_local

            if self.attacker.previous_global_model is None:
                self.attacker.previous_global_model = copy.deepcopy(model)
                return

            candidate_weights = self.attacker.search_candidate_weights(model, proportion=0.1)
            self.attacker.previous_global_model = copy.deepcopy(model)

            if self.attacker.params.handcraft_trigger:
                print("Optimize Trigger:")
                self.attacker.optimize_backdoor_trigger(model, candidate_weights, task, train_handcraft_loader)

            print("Client {}: Inject handcraft filters:".format(self.client_index))
            diff = self.attacker.inject_handcrafted_filters(model, candidate_weights, task, train_handcraft_loader)

            if diff is not None and self.handcraft_rnd % 3 == 1:
                print("Rnd {}: Inject Backdoor FC".format(self.handcraft_rnd))
                self.attacker.inject_handcrafted_neurons(model, candidate_weights, task, diff, train_handcraft_loader)

            self.gen_manager.update_network_parameters(model, self.model)

            self.gen_manager.amplify_genotype_weights(self.model, amplification_factor=1.0)

            # self.gen_manager.amplify_genotype_weights(self.model, amplification_factor=2.0)
            self.syn = self.attacker.synthesizer
            # self.pattern = self.attacker.synthesizer.pattern

    def handcraft_train(self, task):  # F3BA
            self.handcraft_rnd = self.handcraft_rnd + 1

            print("the attacker is {}".format(self.client_index))
            model = self.model
            model.eval()
            train_handcraft_loader, train_loader = self.train_handcraft_loader, self.train_local
            #       fba_attacker ---> attacker
            if self.attacker.previous_global_model is None:
                self.attacker.previous_global_model = copy.deepcopy(model)
                return
            candidate_weights = self.attacker.search_candidate_weights(model, proportion=0.1)
            self.attacker.previous_global_model = copy.deepcopy(model)

            if self.attacker.params.handcraft_trigger:
                print("Optimize Trigger:")
                self.attacker.optimize_backdoor_trigger(model, candidate_weights, task, train_handcraft_loader)

            print("Client {}: Inject handcraft filters:".format(self.client_index))
            print("Inject Candidate Filters:")
            diff = self.attacker.inject_handcrafted_filters(model, candidate_weights, task, train_handcraft_loader)
            if diff is not None and self.handcraft_rnd % 3 == 1:
                print("Rnd {}: Inject Backdoor FC".format(self.handcraft_rnd))
                self.attacker.inject_handcrafted_neurons(model, candidate_weights, task, diff, train_handcraft_loader)
            self.syn = self.attacker.synthesizer
            # self.pattern = self.attacker.synthesizer.pattern

    def search(self):
        self.model.to(self.device)
        # search
        if self.attack and self.args.attack_type == 'f3ba':
            self.handcraft_search(self.task)
        elif self.attack and self.args.attack_type == 'cerp':
            self.current_model = copy.deepcopy(self.model)
            for name, param in self.current_model.named_parameters():
                self.normal_params_variables[name] = self.current_model.state_dict()[name].clone().detach().requires_grad_(
                    False)
            if self.previous_model is not None:
                self.attacker.cerp_trigger(self.model, self.previous_model, self.attacker.synthesizer.pattern,
                                       self.attacker.synthesizer.initial_tensor, self.train_local, self.device)
        elif self.attack and self.args.attack_type == 'a3fl':
            adv_model = copy.deepcopy(self.model)
            self.attacker.search_trigger(adv_model, self.train_local, self.device, self.round)

        # save the global model of the last round
        if self.attack and self.args.attack_type == 'cerp':
            self.previous_model = copy.deepcopy(self.model)

        self.model.train()

        # mask_grad_list = self.get_grad_mask_on_cv(self.task) # neurotoxin_train

        arch_parameters = self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))

        parameters = self.model.parameters()
        weight_params = filter(lambda p: id(p) not in arch_params,
                               parameters)

        optimizer = torch.optim.SGD(
            weight_params,  # model.parameters(),
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay)

        architect = Architect(self.model, self.criterion, self.args, self.device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min)

        local_avg_train_acc = []
        local_avg_train_loss = []
        if self.attack and self.args.attack_type == 'cerp' and self.previous_model is not None:
            self.attack = False
            self.local_search(self.train_local, self.test_local,
                              self.current_model, architect, self.criterion, optimizer)
            self.attack = True
        for epoch in range(self.args.epochs):
            # training
            train_acc, train_obj, train_loss = self.local_search(self.train_local, self.test_local,
                                                                 self.model, architect, self.criterion,
                                                                 optimizer)  # neurotoxin_train, mask_grad_list
            logging.info('client_idx = %d, epoch = %d, local search_acc %f' % (self.client_index, epoch, train_acc))
            local_avg_train_acc.append(train_acc)
            local_avg_train_loss.append(train_loss)

            # # validation
            # with torch.no_grad():
            #     valid_acc, valid_obj, valid_loss = self.local_infer(self.test_local, self.model, self.criterion)
            # logging.info('client_idx = %d, epoch = %d, local valid_acc %f' % (self.client_index, epoch, valid_acc))

            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            logging.info('client_idx = %d, epoch %d lr %e' % (self.client_index, epoch, lr))
        if self.attack:
            local_update = self.get_fl_update(self.model, self.global_model)
            self.attacker.fl_scale_update(local_update)
            self.get_scaled_weight(self.model, local_update)

        weights = self.model.cpu().state_dict()
        alphas = self.model.cpu().arch_parameters()

        logit = []

        return weights, alphas, self.local_sample_number, \
               sum(local_avg_train_acc) / len(local_avg_train_acc), \
               sum(local_avg_train_loss) / len(local_avg_train_loss), logit

    def local_search(self, train_queue, valid_queue, model, architect, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        loss = None

        for step, (input, target) in enumerate(train_queue):

            # logging.info("epoch %d, step %d START" % (epoch, step))
            n = input.size(0)

            # model.set_tau(
            #  self.args.tau_max - self.args.epochs * 1.0 / self.args.epochs * (self.args.tau_max - self.args.tau_min))

            input = input.to(self.device)
            target = torch.Tensor(target.numpy()).long()
            target = target.to(self.device)

            batch = Batch(step, input, target)
            batch = batch.to(Params.device)
            batch_back = self.syn.make_backdoor_batch(batch=batch, test=False, attack=self.attack)

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.to(self.device)
            target_search = target_search.to(self.device)
            batch_search = Batch(step, input_search, target_search)
            batch_search = batch_search.to(self.device)

            if self.attack:
                batch_back_search = self.syn.make_backdoor_batch(batch=batch_search, test=False, attack=self.attack)
                input = batch_back.inputs.to(self.device)
                target = batch_back.labels.to(self.device)
                input_search = batch_back_search.inputs.to(self.device)
                target_search = batch_back_search.labels.to(self.device)
                if self.args.attack_type == "arch":
                    architect.step_back(batch_back, batch_back_search, self.attacker, self.args.lambda_train_regularizer)
                else:
                    architect.step_v2(input, target, input_search, target_search, self.args.lambda_train_regularizer,
                              self.args.lambda_valid_regularizer)
            else:
                architect.step_v2(input, target, input_search, target_search, self.args.lambda_train_regularizer,
                              self.args.lambda_valid_regularizer)

            optimizer.zero_grad()
            # logits = model(input)
            # loss = criterion(logits, target)

            if self.attack:
                logits = model(batch_back.inputs.to(self.device))
                # logging.info(logits)
                target1 = batch_back.labels.to(self.device)
                # loss = criterion(logits, target)
                if self.args.attack_type == "cerp":
                    if self.args. poison_type == "MGDA":
                        loss1 = self.attacker.compute_blind_loss(model, criterion, batch, self.attack, train=False)
                    else:
                        loss1 = criterion(logits, target1)
                    sim_factor = self.attacker.params.alpha_loss
                    # loss = loss1
                    loss = ((1 - sim_factor) * loss1 +
                            sim_factor * model_dist_norm_var(model, self.normal_params_variables, self.device))
                elif self.args.attack_type == "a3fl":
                    # loss = criterion(logits, target)
                    if self.args.poison_type == "MGDA":
                        loss = self.attacker.compute_blind_loss(model, self.criterion, batch, self.attack, train=False)
                    else:
                        loss = criterion(logits, target1)
                else:
                    loss = self.attacker.compute_blind_loss(model, criterion, batch, self.attack, train=False)
            else:
                logits = model(input)
                # logging.info(logits)
                loss = criterion(logits, target)

            # loss.backward(retain_graph=True)  # neurotoxin_train
            # self.apply_grad_mask(model, mask_grad_list)  # neurotoxin_train
            loss.backward()

            parameters = model.arch_parameters()
            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()

            # logging.info("step %d. update weight by SGD. FINISH\n" % step)
            # handcraft_train
            # self.handcraft_train(self.task)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # torch.cuda.empty_cache()

            if step % self.args.report_freq == 0:
                logging.info('client_index = %d, search %03d %e %f %f', self.client_index,
                             step, objs.avg, top1.avg, top5.avg)

        return top1.avg / 100.0, objs.avg / 100.0, loss

    # F3BA in searching phase


    def train(self):
        print("Entering train function")
        self.model.to(self.device)

        # print('start handcrafting')
        if self.attack and self.args.attack_type == 'f3ba':
            self.handcraft_train(self.task)
        elif self.attack and self.args.attack_type == 'cerp':
            self.current_model = copy.deepcopy(self.model)
            # normal_params_variables = dict()
            for name, param in self.current_model.named_parameters():
                self.normal_params_variables[name] = self.current_model.state_dict()[name].clone().detach().requires_grad_(
                    False)
            if self.previous_model is not None:
                self.attacker.cerp_trigger(self.model, self.previous_model, self.attacker.synthesizer.pattern,
                                       self.attacker.synthesizer.initial_tensor, self.train_local, self.device)
        elif self.attack and self.args.attack_type == 'a3fl':
            adv_model = copy.deepcopy(self.model)
            self.attacker.search_trigger(adv_model, self.train_local, self.device, self.round)

        # save the global model of the last round

        print('start training')
        
        self.model.train()

        parameters = self.model.parameters()

        optimizer = torch.optim.SGD(
            parameters,  # model.parameters(),
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min)
        # self.handcraft_train(self.task)  # F3BA
        local_avg_train_acc = []
        local_avg_train_loss = []
        if self.attack and self.args.attack_type == 'cerp' and self.previous_model is not None:
            self.attack = False
            self.local_train(self.train_local, self.test_local,
                             self.current_model, self.criterion,
                             optimizer)
            self.attack = True

        for epoch in range(self.args.epochs):
            # training
            train_acc, train_obj, train_loss = self.local_train(self.train_local, self.test_local,
                                                                self.model, self.criterion,
                                                                optimizer)
            logging.info('client_idx = %d, local train_acc %f' % (self.client_index, train_acc))
            local_avg_train_acc.append(train_acc)
            local_avg_train_loss.append(train_loss)

            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            logging.info('client_idx = %d, epoch %d lr %e' % (self.client_index, epoch, lr))
        if self.attack and self.args.attack_type == 'cerp':
            self.previous_model = copy.deepcopy(self.model)
        logit = []
        if self.args.defense_method == 'FEDRAD':
            logit = self.distillation_logit()
        if self.attack:
            logging.info("------------------------------------")
            logging.info(self.round)
            local_update = self.get_fl_update(self.model, self.global_model)
            self.attacker.fl_scale_update(local_update)
            self.get_scaled_weight(self.model, local_update)
        elif self.args.defense_method == 'ESFL':
            self.local_dp()
        weights = self.model.cpu().state_dict()


        # adaptive_distillation


        return weights, self.local_sample_number, \
               sum(local_avg_train_acc) / len(local_avg_train_acc), \
               sum(local_avg_train_loss) / len(local_avg_train_loss),logit

    def local_train(self, train_queue, valid_queue, model, criterion, optimizer):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        batch_id = 0

        for step, (input, target) in enumerate(train_queue):
            # logging.info("epoch %d, step %d START" % (epoch, step))
            model.train()
            n = input.size(0)
            target = torch.Tensor(target.numpy()).long()
            input = input.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            model.to(self.device)
            # logits, logits_aux = model(input)
            outputs = model(input)
            if isinstance(outputs, tuple):
                logits, logits_aux = outputs
            else:
                logits = outputs
            target = torch.Tensor(target.cpu().numpy()).long()
            target = target.to(self.device)
            batch = Batch(step, input, target)
            batch = batch.to(Params.device)
            batch_back = self.syn.make_backdoor_batch(batch=batch, test=False, attack=self.attack)

            if self.attack:
                # logits, logits_aux = model(batch_back.inputs.to(self.device))
                outputs = model(batch_back.inputs.to(self.device))
                if isinstance(outputs, tuple):
                    logits, logits_aux = outputs
                else:
                    logits = outputs

                # logging.info(logits)
                target1 = batch_back.labels.to(self.device)

                # baseline attack
                # logits, logits_aux = model(input)
                # target = batch_back.labels.to(self.device)
                # loss = criterion(logits, target)

                if self.args.stage == "train":
                    loss = self.attacker.compute_blind_loss(model, self.criterion, batch, self.attack, train=True)
                    if self.args.attack_type == "cerp":
                        if self.args.poison_type == 'MGDA':
                            loss1 = self.attacker.compute_blind_loss(model, criterion, batch, self.attack, train=True)
                        else:
                            loss1 = criterion(logits, target1)
                        # loss = loss1
                        sim_factor = self.attacker.params.alpha_loss
                        loss = (1 - sim_factor) * loss1 + sim_factor * model_dist_norm_var(model, self.normal_params_variables, self.device)
                    elif self.args.attack_type == "a3fl":
                        # loss = criterion(logits, target1)
                        if self.args.poison_type == 'MGDA':
                            loss = self.attacker.compute_blind_loss(model, criterion, batch, self.attack, train=True)
                        else:
                            loss = criterion(logits, target1)
                    else:
                        loss = self.attacker.compute_blind_loss(model, criterion, batch, self.attack, train=True)



                else:
                    # loss = self.attacker.compute_blind_loss(model, self.criterion, batch, self.attack, train=False)
                    if self.args.attack_type == "cerp":
                        # loss1 = criterion(logits, target1)
                        if self.args.poison_type == 'MGDA':
                            loss1 = self.attacker.compute_blind_loss(model, criterion, batch, self.attack, train=False)
                        else:
                            loss1 = criterion(logits, target1)
                        sim_factor = self.attacker.params.alpha_loss

                        loss = (1 - sim_factor) * loss1 + sim_factor * model_dist_norm_var(model, self.normal_params_variables, self.device)
                    elif self.args.attack_type == "a3fl":

                        if self.args.poison_type == 'MGDA':
                            loss = self.attacker.compute_blind_loss(model, self.criterion, batch,
                                                                    self.attack, train=False)
                        else:
                            loss = criterion(logits, target1)

                    else:
                        loss = self.attacker.compute_blind_loss(model, criterion, batch, self.attack, train=False)
            else:
                # logits, logits_aux = model(input)
                outputs = model(batch_back.inputs.to(self.device))
                if isinstance(outputs, tuple):
                    logits, logits_aux = outputs
                else:
                    logits = outputs

                # logging.info(logits)
                loss = criterion(logits, target)
            
            if self.args.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += self.args.auxiliary_weight * loss_aux
            loss.backward()
            parameters = model.parameters()

            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()
            # logging.info("step %d. update weight by SGD. FINISH\n" % step)
            # self.handcraft_train(self.task)  # F3BA
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            batch_id = batch_id + 1

            # torch.cuda.empty_cache()
            if step % self.args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg, loss



