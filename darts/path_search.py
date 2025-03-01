import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.operations import *
from darts.genotypes import PRIMITIVES, Genotype

def set_weights_biases_to_module(module, weights, biases):
    if weights is None:
        weights = [None]
    if biases is None:
        biases = [None]

    weight_idx = 0
    bias_idx = 0

    if len(weights) == 1:
        if hasattr(module, 'weight') and module.weight is not None and weights[0] is not None:
            module.weight.data = weights[0]
    if len(biases) == 1:
        if hasattr(module, 'bias') and module.bias is not None and biases[0] is not None:
            module.bias.data = biases[0]

    if len(weights) > 1 or len(biases) > 1:
        for child_name, child in module.named_children():
            if hasattr(child, 'weight') and child.weight is not None and weights[weight_idx] is not None:
                child.weight.data = weights[weight_idx]
                weight_idx += 1

            if hasattr(child, 'bias') and child.bias is not None and biases[bias_idx] is not None:
                child.bias.data = biases[bias_idx]
                bias_idx += 1


def get_weights_biases_from_module(module):
    weights = []
    biases = []

    for child_name, child in module.named_children():
        if hasattr(child, 'weight') and child.weight is not None:
            weights.append(child.weight.data)
        if hasattr(child, 'bias') and child.bias is not None:
            biases.append(child.bias.data)

        for grandchild_name, grandchild in child.named_children():
            if hasattr(grandchild, 'weight') and grandchild.weight is not None:
                weights.append(grandchild.weight.data)
            if hasattr(grandchild, 'bias') and grandchild.bias is not None:
                biases.append(grandchild.bias.data)

    return weights, biases


def extract_weights_biases(op):
    weights_list = []
    biases_list = []

    if hasattr(op, 'weight'):
        weights_list.append(op.weight.data)
        if hasattr(op, 'bias') and op.bias is not None:
            biases_list.append(op.bias.data)
        else:
            biases_list.append(None)

    if not weights_list or not biases_list:
        for name, child in op.named_children():
            if isinstance(child, (DilConv, SepConv, FactorizedReduce)):
                child_weights, child_biases = get_weights_biases_from_module(child)
                if child_weights is not None:
                    weights_list.append(child_weights)
                else:
                    weights_list.append(None)

                if child_biases is not None:
                    biases_list.append(child_biases)
                else:
                    biases_list.append(None)

    return weights_list, biases_list


def transfer_weights_from_monitored_op_to_mixed_op(monitored_op, mixed_op, primitive_name):
    idx = PRIMITIVES.index(primitive_name)
    monitored_state_dict = monitored_op.op.state_dict()
    mixed_op._ops[idx].load_state_dict(monitored_state_dict)


class GenotypeManager:
    def __init__(self, network):
        self.network = network
        self.genotype = None

        self.gene_normal = None #
        self.gene_reduce = None #

        # Save or load models from files
        self.save_all_name = 'network_all.pth'
        self.save_weight_name = 'network_weight.pth'
        self.load_all_name = 'network_all.pth'
        self.load_weight_name = 'network_weight.pth'

    def set_genotype(self, genotype):
        self.genotype = genotype
        self.gene_normal = genotype.normal #
        self.gene_reduce = genotype.reduce #

    def set_save_all_name(self, save_all_name):
        self.save_all_name = save_all_name

    def set_save_weight_name(self, save_weight_name):
        self.save_weight_name = save_weight_name

    def set_load_all_name(self, load_all_name):
        self.load_all_name = load_all_name

    def set_load_weight_name(self, load_weight_name):
        self.load_weight_name = load_weight_name

    def save_genotype_model(self):
        monitored_network = self.modify_network_with_genotype(self.network)
        # Save the whole model to the file
        torch.save(monitored_network, self.save_all_name)
        torch.save(monitored_network.state_dict(), self.save_weight_name)

    def load_model_all(self, init_channels, classes, layers, criterion, genotype):
        model = MonitoredNetwork(init_channels, classes, layers, criterion, genotype)
        model.load(self.load_all_name)


    def load_model_weight(self, init_channels, classes, layers, criterion, genotype):
        model = MonitoredNetwork(init_channels, classes, layers, criterion, genotype)
        model.load_state_dict(self.load_weight_name)

    def random_sample_genotype(self):
        def _sample(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.network._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -W[x].max())[0:2]
                for j in edges:
                    k_best = W[j].argmax().item()
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        with torch.no_grad():
            weights_normal = F.softmax(self.network.alphas_normal, dim=-1).data.cpu().numpy()
            weights_reduce = F.softmax(self.network.alphas_reduce, dim=-1).data.cpu().numpy()

            self.gene_normal = _sample(weights_normal)
            self.gene_reduce = _sample(weights_reduce)

            concat = range(2 + self.network._steps - self.network._multiplier, self.network._steps + 2)
            self.genotype = Genotype(normal=self.gene_normal, normal_concat=concat, reduce=self.gene_reduce, reduce_concat=concat)

    def get_weights_from_genotype(self, gene, alphas_size):
        weights = torch.zeros(alphas_size)
        for (op, index) in gene:
            op_index = PRIMITIVES.index(op)
            weights[index][op_index] = 1.0
        return weights

    def get_weights_from_current_architecture(self, network):
        with torch.no_grad():
            weights_normal = self.get_weights_from_genotype(self.gene_normal, network.alphas_normal.size())
            weights_reduce = self.get_weights_from_genotype(self.gene_reduce, network.alphas_reduce.size())
        return weights_normal, weights_reduce

    def modify_network_with_genotype(self, network):
        # Create a MonitoredNetwork with the same configurations as the original network
        monitored_network = MonitoredNetwork(network._C, network._num_classes, network._layers, network._criterion,
                                             self.genotype)

        for (name_param_net, param_net), (name_param_monitored, param_monitored) in zip(network.named_parameters(),
                                                                                        monitored_network.named_parameters()):
            if name_param_net == name_param_monitored:
                param_monitored.data.copy_(param_net.data)

        return monitored_network

    def amplify_genotype_weights(self, network, amplification_factor=1.5):
        with torch.no_grad():
            # Amplify weights for normal cells
            for op, j in self.genotype.normal:
                idx = PRIMITIVES.index(op)
                network.alphas_normal[j][idx] *= amplification_factor

            # Amplify weights for reduction cells
            for op, j in self.genotype.reduce:
                idx = PRIMITIVES.index(op)
                network.alphas_reduce[j][idx] *= amplification_factor

    def transfer_weights(self, MonitoredOp, moduleB, primitive_name):

        handcrafted_op = MonitoredOp._ops
        # Get the weights and biases from submoduleA
        handcrafted_weight, handcrafted_bias = extract_weights_biases(handcrafted_op)
        op = moduleB.get_op_by_primitive(primitive_name)

        set_weights_biases_to_module(op, handcrafted_weight, handcrafted_bias)



    def update_network_parameters(self, handcrafted_model, original_network):
        for original_cell, handcrafted_cell in zip(original_network.cells, handcrafted_model.cells):
            chosen_primitive_name = handcrafted_cell._ops[0].primitive_name
            idx = PRIMITIVES.index(chosen_primitive_name)


            # Map the weights of ops
            for original_op, handcrafted_op in zip(original_cell._ops, handcrafted_cell._ops):
                # The list of MonitoredOp and MixedOp respectively
                primitive_name = handcrafted_op.primitive_name
                self.transfer_weights(handcrafted_op, original_op, primitive_name)




class MonitoredNetwork(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, genotype, stem_multiplier=3):
        super(MonitoredNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        # C_prev_prev, C_prev = C_curr, C_curr
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            if reduction:
                cell_genotype = genotype.reduce
            else:
                cell_genotype = genotype.normal

            cell = MonitoredCell(cell_genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, C_curr  # Adjusted based on the new MonitoredCell output

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # self.weights_normal = None
        # self.weights_reduce = None

    """
    def set_weights(self, weights_normal, weights_reduce):
        self.weights_normal = weights_normal
        self.weights_reduce = weights_reduce
    """
    def features(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1)
            else:
                s0, s1 = s1, cell(s0, s1)
        return s1

    def first_activations(self, input):
        s0 = s1 = self.stem(input)
        return s1

    def final_activations(self, input):
        return self.features(input)

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                s0, s1 = s1, cell(s0, s1)
            else:
                s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits



class MonitoredCell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(MonitoredCell, self).__init__()
        self.reduction = reduction
        self.genotype = genotype

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        self._ops = nn.ModuleList()
        for (primitive, _) in genotype:
            stride = 1
            op = MonitoredOp(C, stride, primitive)
            self._ops.append(op)

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i, (primitive, prev_input_index) in enumerate(self.genotype):
            h = states[prev_input_index]
            op = self._ops[i]
            s = op(h)
            states.append(s)

            # Return the last state
        return states[-1]

class MonitoredOp(nn.Module):
    def __init__(self, C, stride, primitive_name):
        super(MonitoredOp, self).__init__()
        self.primitive_name = primitive_name
        self._ops = self._build_op(C, stride, primitive_name)

    def _build_op(self, C, stride, primitive_name):
        if 'pool' in primitive_name:
            return nn.Sequential(OPS[primitive_name](C, stride, False),
                                 nn.BatchNorm2d(C, affine=False))
        return OPS[primitive_name](C, stride, False)



    def forward(self, x):
        return self._ops(x)
