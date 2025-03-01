import logging
import time
import math
import torch
import wandb
import numpy as np
from torch import optim, nn


from darts import genotypes, utils, genotype_attention, tiny_geno_genotypes, cifar_geno_genotype
from darts.model import NetworkCIFAR
from darts.model_genospace_cifar import NetworkCIFAR_GENO_CIFAR
from darts.model_tiny_geno import NetworkGENOTINY
# from darts.model_search import Network
from darts.model_attention import NetworkATTENTION
# from darts.tiny_geno_model_search import Network
from darts.model_search_genospace_cifar import Network_GENO_CIFAR
from torch import nn
from torchsummaryX import summary
from utils.parameters import Params
from attack_f3ba import Attack_F3BA
from synthesizers.pattern_synthesizer import *
from tasks.batch import Batch
from data_preprocessing.data_loader import  get_dataloader
from models.resnet import resnet18
from datetime import datetime

class FedNASAggregator(object):
    def __init__(self, train_global, test_global, all_train_data_num, client_num, device, args, attack, attacker):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num
        self.client_num = client_num
        self.device = device
        self.args = args
        self.model = self.init_model()
        self.model_dict = dict()
        self.arch_dict = dict()
        self.sample_num_dict = dict()
        self.train_acc_dict = dict()
        self.train_loss_dict = dict()
        self.train_acc_avg = 0.0
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0
        self.back_test_acc_avg = 0.0
        self.back_test_loss_avg = 0.0
        self.logit_dict = dict()  # adaptive_distillation
        self.syn_dict = dict()  # make pattern
        self.participated_clients = list(range(0, client_num))
        self.chosen_ids = []
        self.optimizer = optim.SGD(filter(lambda layer: layer.requires_grad, self.model.parameters()),
                                   lr=0.0001,
                                   momentum=0.9)

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(0, self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.best_accuracy = 0
        self.best_accuracy_different_cnn_counts = dict()
        self.wandb_table = wandb.Table(columns=["Epoch", "Searched Architecture"])
        self.attack = attack
        self.attacker = attacker
        self.syn = attacker.synthesizer
        # adaptive_distillation
        # self.batch_idxs = self.batch_idx_for_client()
        # self.dataloader_distillation = self.bacth_for_client()


        
    def select_participated_clients(self, size, attacker_num):
        n_chosen = self.args.participant_number
        candidate_ids = list()
        attackers = list(range(0, attacker_num))
        logging.info("size: %d" % size)
        logging.info("number of attackers: %d" % attacker_num)
        logging.info("set of attackers: {}".format(attackers))

        print(size)
        print(attackers)
        for c in range(0, size-1):
            if c not in attackers:
                candidate_ids.append(c)
        selected_benigns = np.random.choice(candidate_ids, n_chosen - len(attackers), replace=False)
        # return the list of selected clients (adversary + benign)
        participated_clients = attackers + list(selected_benigns)
        self.participated_clients = participated_clients
        print(participated_clients)
        logging.info("clients: {}".format(participated_clients))

        return self.participated_clients

    def init_model(self):
        if self.args.search_space == 'TINYGENO':
            from darts.tiny_geno_model_search import Network
        else:
            from darts.model_search import Network
        criterion = nn.CrossEntropyLoss().to(self.device)
        num_classes = 10
        if self.args.dataset == 'cifar10':
            num_classes = 10
        elif self.args.dataset == 'tiny':
            num_classes = 80
        if self.args.stage == "search":
            if self.args.search_space == "DARTS" or self.args.search_space == "TINYGENO":
                model = Network(self.args.init_channels, num_classes, self.args.layers, criterion, self.device)
            elif self.args.search_space == "GENOSPACE":
                model = Network_GENO_CIFAR(self.args.init_channels, num_classes, self.args.layers, criterion, self.device)
        # model with fixed architecture
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

        # model.to(self.device)
        # summary(model, torch.zeros(2, 3, 32, 32).to(self.device))
        # # summary(model, (3, 224, 224), batch_size=64)
        return model

    def get_model(self):
        return self.model

    def get_syn(self):
        return self.syn

    def add_local_trained_result(self, index, model_params, arch_params, sample_num, train_acc, train_loss, logit, syn):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.arch_dict[index] = arch_params
        self.sample_num_dict[index] = sample_num
        self.train_acc_dict[index] = train_acc
        self.train_loss_dict[index] = train_loss
        self.flag_client_model_uploaded_dict[index] = True
        self.logit_dict[index] = logit
        self.chosen_ids.append(index)
        if syn != None:
            self.syn_dict[index] = syn

    # used for making backdoor batch
    def make_synthesizer_pattern(self):
        if len(self.syn_dict) == 0:
            return
        else:
            mix_pattern = 0
            for syn in self.syn_dict.values():
                mix_pattern += syn.pattern
            self.syn.pattern = mix_pattern / len(self.syn_dict)

    def check_whether_all_receive(self):

        for idx in range(0, self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(0, self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False


        # return True
        """
        for idx in self.participated_clients:
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in self.participated_clients:
            self.flag_client_model_uploaded_dict[idx] = False
        """

        return True



    def model_global_norm(self, model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    # CRFL
    def clip_weight_norm(self, model, clip):
        total_norm = self.model_global_norm(model)
        logging.info("total_norm: " + str(total_norm) + "clip_norm: "+str(clip))
        max_norm = clip
        clip_coef = max_norm / (total_norm + 1e-6)
        current_norm = total_norm
        if total_norm > max_norm:
            for name, layer in model.named_parameters():
                layer.data.mul_(clip_coef)
            current_norm = self.model_global_norm(model)
            logging.info("clip~~~ norm after clipping: "+ str(current_norm) )
        return current_norm

    # CRFL
    def dp_noise(self,param, sigma):
        noised_layer = torch.normal(0, sigma, size=param.size()).cuda()
        noised_layer = noised_layer.to(self.device)
        return noised_layer

    # FEDRAD
    def get_median_logits(self, batch_index: int, batch: Batch) -> Batch:
        ensembled_batch = batch.clone()
        current_batch_logits = []
        for client_logits in self.logit_dict.values():
            current_batch_logit = client_logits[batch_index]
            current_batch_logits.append(current_batch_logit)
        current_batch_logits_tensor = torch.stack(current_batch_logits)
        median_logit = torch.median(current_batch_logits_tensor, dim=0).values
        ensembled_batch.labels = median_logit

        return ensembled_batch

    # FEDRAD
    def get_median_counts(self, batch_index: int) -> list:
        indice_counts = []
        with torch.no_grad():
            batch_logits = [self.logit_dict[client_id][batch_index] for client_id in self.chosen_ids]
            batch_logits_tensor = torch.stack(batch_logits)
            _, indices = torch.median(batch_logits_tensor, dim=0)
            indices = indices.view(-1).tolist()
            for i in range(len(self.chosen_ids)):
                indice_counts.append(indices.count(i))
        return indice_counts

    # FEDRAD
    def get_median_scores(self):
        self.model.train()
        median_counts = [0 for i in range(len(self.chosen_ids))]
        for i, (input, target) in enumerate(self.train_global):
            input = input.to(self.device)
            target = target.to(self.device)
            batch = Batch(i, input, target)
            median_count = self.get_median_counts(i)
            median_counts = [x + y for x, y in zip(median_counts, median_count)]
        total_counts = sum(median_counts)
        self.pts = [(med_count / total_counts) for med_count in median_counts]
        return self.pts

    # FEDRAD
    def adaptive_distillation(self):
        self.model.train()
        for i, (inputs, targets) in enumerate(self.train_global):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch = Batch(i, inputs, targets).to(self.device)
            batch = self.get_median_logits(i, batch)
            batch.labels = batch.labels.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(batch.inputs)
            if isinstance(output, tuple):
                predicted_labels = output[0]
            else:
                predicted_labels = output
            # predicted_labels, _ = self.model(batch.inputs)
            kl_div_loss = nn.KLDivLoss(reduction='batchmean')(
                predicted_labels.softmax(dim=-1).log(), batch.labels.softmax(dim=-1)
            )
            kl_div_loss.backward()
            self.optimizer.step()

    """
    def get_batch(self, batch_id, data) -> Batch:
        inputs, labels = data
        batch = Batch(batch_id, inputs, labels)
        return batch.to(Params.device)
        """

    # SSD
    def calculate_difference_norm(self, dict1, dict2):
        diffs = [torch.flatten(dict1[key] - dict2[key]) for key in dict1.keys()]
        return torch.norm(torch.cat(diffs))

    # SSD
    def suspicious_subgroup_detection(self, X_L, global_model, rho, T):
        D = []
        for xi in X_L:
            diffs = [torch.flatten(xi[key] - global_model[key]) for key in xi.keys()]
            di = torch.norm(torch.cat(diffs))
            D.append(di)
        D = torch.tensor(D)

        std_D = torch.std(D)
        xi_t = rho / T * torch.sum(std_D)

        M = []

        while torch.max(D) > xi_t:
            idx = torch.argmax(D)
            X_L[idx] = global_model
            D[idx] = 0
            M.append(idx.item())

            D[idx] = self.calculate_difference_norm(X_L[idx], global_model)
            xi_t = rho / T * torch.sum(torch.std(D))

        return M, X_L

    # SSD
    def aggregate_global_model_from_list(self, model_list, surrogate_params_list):
        total_data_num = sum([item[0] for item in model_list])
        global_params = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in surrogate_params_list[0].items()}
        for idx, params in enumerate(surrogate_params_list):
            sample_num1 = model_list[2 * idx][0]
            sample_num2 = model_list[2 * idx + 1][0] if 2 * idx + 1 < len(
                model_list) else 0
            combined_sample_num = sample_num1 + sample_num2
            logging.info(
                f'Index: {idx}, Sample Num 1: {sample_num1}, '
                f'Sample Num 2: {sample_num2}, Combined Sample Num: {combined_sample_num}')

            w = combined_sample_num / total_data_num
            logging.info(f'Weight (w): {w}')
            for k, v in params.items():
                # global_params[k] = global_params[k].float()
                global_params[k] += v * w
        # logging.info(f'Final Global Params: {global_params}')

        return global_params

    def aggregate(self,round_idx):

        averaged_weights = self.__aggregate_weight()
        self.model.load_state_dict(averaged_weights)

        if self.args.defense_method == 'CRFL':
            start_time = datetime.now()
            
            dynamic_thres = self.args.current_round * 0.25 + 125
            param_clip_thres = 130
            if dynamic_thres < param_clip_thres:
                param_clip_thres = dynamic_thres
            current_norm = self.clip_weight_norm(self.model, param_clip_thres)
            logging.info("current norm: " + str(current_norm))
            
            end_time = datetime.now() 
            execution_time = end_time - start_time
            total_seconds = execution_time.total_seconds()
            wandb.log({"total_seconds": total_seconds, "Round": round_idx})

        if self.args.stage == "search":
            averaged_alphas = self.__aggregate_alpha()
            self.__update_arch(averaged_alphas)
            return averaged_weights, averaged_alphas
        else:
            return averaged_weights

    def __update_arch(self, alphas):
        logging.info("update_arch. server.")
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    def __aggregate_weight(self):
        logging.info("################aggregate weights############")
        start_time = time.time()
        model_list = []
        # for idx in self.participated_clients:
        for idx in range(self.client_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
        (num0, averaged_params) = model_list[0]

        # vanilla version
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / self.all_train_data_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        if self.args.defense_method == "SSD":
            pairwise_averaged_params_list = []
            logging.info("################SSD############")

            for i in range(0, len(model_list) - 1, 2):
                (num1, params1) = model_list[i]
                (num2, params2) = model_list[i + 1]

                pairwise_averaged_params = {}
                for k in params1.keys():
                    w1 = num1 / (num1 + num2)
                    w2 = num2 / (num1 + num2)
                    pairwise_averaged_params[k] = params1[k] * w1 + params2[k] * w2

                pairwise_averaged_params_list.append(pairwise_averaged_params)
            rho = 1.2
            T = 100
            M, surrogate_params_list = self.suspicious_subgroup_detection(pairwise_averaged_params_list, averaged_params, rho, T)

            averaged_params = self.aggregate_global_model_from_list(model_list, surrogate_params_list)
        elif self.args.defense_method == 'FEDRAD':
            # print(self.pts)
            # print(self.chosen_ids)
            total_prop = 0
            for pt, id in zip(self.pts, self.chosen_ids):
                # print("pts id: ", id)
                # print("pts pt: ", pt)
                total_prop = total_prop + self.sample_num_dict[id] * pt
            w_dict = {}
            for pt, id in zip(self.pts, self.chosen_ids):
                w_dict[id] = self.sample_num_dict[id] * pt / total_prop

            for k in averaged_params.keys():
                j = 0
                for i in range(0, len(model_list)):
                    local_sample_number, local_model_params = model_list[i]
                    # print("local_model_params id: ", i)
                    if j == 0:
                        averaged_params[k] = local_model_params[k] * w_dict[i]
                    else:
                        averaged_params[k] += local_model_params[k] * w_dict[i]
                    j += 1


        # protection version
        # for k in averaged_params.keys():
        #     for i in range(1, len(model_list)):
        #         local_sample_number, local_model_params = model_list[i]
        #         ex_local_sample_number, ex_local_model_params = model_list[0]
        #         w = local_sample_number / (self.all_train_data_num - ex_local_sample_number)
        #         if i == 1:
        #             averaged_params[k] = local_model_params[k] * w
        #         else:
        #             averaged_params[k] += local_model_params[k] * w

        # protection version

        # clear the memory cost
        model_list.clear()
        del model_list
        self.model_dict.clear()
        end_time = time.time()
        logging.info("aggregate weights time cost: %d" % (end_time - start_time))
        return averaged_params

    def __aggregate_alpha(self):
        logging.info("################aggregate alphas############")
        start_time = time.time()
        alpha_list = []
        # for idx in range(self.client_num):
        for idx in self.participated_clients:
            alpha_list.append((self.sample_num_dict[idx], self.arch_dict[idx]))

        (num0, averaged_alphas) = alpha_list[0]
        # vanilla version
        for index, alpha in enumerate(averaged_alphas):
            for i in range(0, len(alpha_list)):
                local_sample_number, local_alpha = alpha_list[i]
                w = local_sample_number / self.all_train_data_num
                if i == 0:
                    averaged_alphas[index] = local_alpha[index] * w
                else:
                    averaged_alphas[index] += local_alpha[index] * w

        # protection version
        # for index, alpha in enumerate(averaged_alphas):
        #     for i in range(1, len(alpha_list)):
        #         local_sample_number, local_alpha = alpha_list[i]
        #         ex_local_sample_number, ex_local_alpha = alpha_list[0]
        #         w = local_sample_number / (self.all_train_data_num - ex_local_sample_number)
        #         if i == 1:
        #             averaged_alphas[index] = local_alpha[index] * w
        #         else:
        #             averaged_alphas[index] += local_alpha[index] * w

        end_time = time.time()
        logging.info("aggregate alphas time cost: %d" % (end_time - start_time))
        return averaged_alphas

    def statistics(self, round_idx):
        index = 0
        for syn in self.syn_dict.values():
            pattern_tensor = syn.pattern_tensor
            logging.info('pattern_tensor = %s', str(pattern_tensor))
            wandb.log({"genotype": str(pattern_tensor), "round_idx": index})
            index += 1

        # train acc
        train_acc_list = self.train_acc_dict.values()
        self.train_acc_avg = sum(train_acc_list) / len(train_acc_list)
        logging.info('Round {:3d}, Average Train Accuracy {:.3f}'.format(round_idx, self.train_acc_avg))
        wandb.log({"Train Accuracy": self.train_acc_avg, "Round": round_idx})
        # train loss
        _train_loss_list = self.train_loss_dict.values()
        train_loss_list = []
        for value in _train_loss_list:
            _value = value.to(self.device)
            train_loss_list.append(_value)
        train_loss_avg = sum(train_loss_list) / len(train_loss_list)
        logging.info('Round {:3d}, Average Train Loss {:.3f}'.format(round_idx, train_loss_avg))
        wandb.log({"Train Loss": train_loss_avg, "Round": round_idx})

        # test acc
        logging.info('Round {:3d}, Average Validation Accuracy {:.3f}'.format(round_idx, self.test_acc_avg))
        wandb.log({"Validation Accuracy": self.test_acc_avg, "Round": round_idx})
        # test loss
        logging.info('Round {:3d}, Average Validation Loss {:.3f}'.format(round_idx, self.test_loss_avg))
        wandb.log({"Validation Loss": self.test_loss_avg, "Round": round_idx})

        if round_idx >= self.args.attack_round:
            # test acc
            logging.info('Round {:3d}, Average Backdoored Validation Accuracy {:.3f}'.format(round_idx, self.back_test_acc_avg))
            wandb.log({"ASR": self.back_test_acc_avg, "Round": round_idx - self.args.attack_round})
            # test loss
            logging.info('Round {:3d}, Average Backdoored Validation Loss {:.3f}'.format(round_idx, self.back_test_loss_avg))
            wandb.log({"Backdoored Validation Loss": self.back_test_loss_avg, "Round": round_idx - self.args.attack_round})

        logging.info("search_train_valid_acc_gap %f" % (self.train_acc_avg - self.test_acc_avg))
        wandb.log({"search_train_valid_acc_gap": self.train_acc_avg - self.test_acc_avg, "Round": round_idx})

        logging.info("search_train_backdoored_valid_acc_gap %f" % (self.train_acc_avg - self.back_test_acc_avg))
        wandb.log({"search_train_backdoored_valid_acc_gap": self.train_acc_avg - self.back_test_acc_avg, "Round": round_idx})



    def infer(self, round_idx):
        self.model.eval()
        self.model.to(self.device)
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            start_time = time.time()
            test_correct = 0.0
            test_loss = 0.0
            test_sample_number = 0.0
            back_test_correct = 0.0
            back_test_loss = 0.0
            back_test_sample_number = 0.0
            test_data = self.test_global
            # loss
            criterion = nn.CrossEntropyLoss().to(self.device)
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_data):

                    x = x.to(self.device)
                    target = torch.Tensor(target.cpu().numpy()).long()
                    target = target.to(self.device)
                    batch = Batch(batch_idx, x, target)
                    batch = batch.to(Params.device)
                    # new version
                    if self.args.attack_type != "constrain":
                        self.make_synthesizer_pattern()

                    batch_back = self.syn.make_backdoor_batch(batch=batch, test=True, attack=self.attack)
                    back_target = batch_back.labels
                    back_target = back_target.to(self.device)
                    pred = self.model(x)
                    pred_back = self.model(batch_back.inputs.to(self.device))
                    if isinstance(pred, tuple):
                        logits = pred[0]
                    else:
                        logits = pred
                    if isinstance(pred_back, tuple):
                        logits_back = pred_back[0]
                    else:
                        logits_back = pred_back

                    loss = criterion(logits, target)
                    back_loss = criterion(logits_back, back_target)
                    _, predicted = torch.max(logits, 1)
                    _, predicted_back = torch.max(logits_back, 1)
                    correct = predicted.eq(target).sum()
                    back_correct = predicted_back.eq(back_target).sum()

                    test_correct += correct.item()
                    test_loss += loss.item() * target.size(0)
                    test_sample_number += target.size(0)

                    back_test_correct += back_correct.item()
                    back_test_loss += back_loss.item() * back_target.size(0)
                    back_test_sample_number += back_target.size(0)
            logging.info("server test. round_idx = %d, test_loss = %s" % (round_idx, test_loss))
            logging.info("server backdoor_test. round_idx = %d, test_loss = %s" % (round_idx, back_test_loss))

            self.test_acc_avg = test_correct / test_sample_number
            self.test_loss_avg = test_loss

            self.back_test_acc_avg = back_test_correct / back_test_sample_number
            self.back_test_loss_avg = back_test_loss

            if self.args.defense_method == 'CRFL':
                for name, param in self.model.state_dict().items():
                    noise = self.dp_noise(param, self.args.sigma)
                    param.float().add_(noise.float())
            if self.args.defense_method == "FEDRAD":
                self.adaptive_distillation()

            end_time = time.time()
            logging.info("server_infer time cost: %d" % (end_time - start_time))
            if self.args.stage == "train":
                torch.save(self.model,  str(self.args.dataset) + "_"'train_all.pth')
                torch.save(self.model.state_dict(),  str(self.args.dataset) + "_"'train_weights.pth')
        return self.syn

    def record_model_global_architecture(self, round_idx):
        # save the model architecture
        torch.save(self.model, str(self.args.search_space) +
                       "_" + str(self.args.dataset) + "_"+'search_space_all.pth')
        torch.save(self.model.state_dict(),str(self.args.search_space) +
                       "_" + str(self.args.dataset) + "_" + 'search_space_weights.pth')


        genotype, normal_cnn_count, reduce_cnn_count = self.model.genotype()
        cnn_count = normal_cnn_count + reduce_cnn_count
        wandb.log({"cnn_count": cnn_count, "Round": round_idx})

        logging.info("(n:%d,r:%d)" % (normal_cnn_count, reduce_cnn_count))
        logging.info('genotype = %s', genotype)
        wandb.log({"genotype": str(genotype), "round_idx": round_idx})

        self.wandb_table.add_data(str(round_idx), str(genotype))
        wandb.log({"Searched Architecture": self.wandb_table})

        # save the cnn architecture according to the CNN count
        cnn_count = normal_cnn_count * 10 + reduce_cnn_count
        wandb.log({"searching_cnn_count(%s)" % cnn_count: self.test_acc_avg, "epoch": round_idx})
        if cnn_count not in self.best_accuracy_different_cnn_counts.keys():
            self.best_accuracy_different_cnn_counts[cnn_count] = self.test_acc_avg
            summary_key_cnn_structure = "best_acc_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
            wandb.run.summary[summary_key_cnn_structure] = self.test_acc_avg

            summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (
                normal_cnn_count, reduce_cnn_count)
            wandb.run.summary[summary_key_best_cnn_structure] = round_idx
        else:
            if self.test_acc_avg > self.best_accuracy_different_cnn_counts[cnn_count]:
                self.best_accuracy_different_cnn_counts[cnn_count] = self.test_acc_avg
                summary_key_cnn_structure = "best_acc_for_cnn_structure(n:%d,r:%d)" % (
                    normal_cnn_count, reduce_cnn_count)
                wandb.run.summary[summary_key_cnn_structure] = self.test_acc_avg

                summary_key_best_cnn_structure = "epoch_of_best_acc_for_cnn_structure(n:%d,r:%d)" % (
                    normal_cnn_count, reduce_cnn_count)
                wandb.run.summary[summary_key_best_cnn_structure] = round_idx

        if self.test_acc_avg > self.best_accuracy:
            self.best_accuracy = self.test_acc_avg
            wandb.run.summary["best_valid_accuracy"] = self.best_accuracy
            wandb.run.summary["epoch_of_best_accuracy"] = round_idx

    # return best genotype to attacker
    def return_genotype(self):
        genotype, normal_cnn_count, reduce_cnn_count = self.model.genotype()
        return genotype