import argparse
import logging
import os
import socket
from utils.parameters import Params
import numpy as np
import psutil
import setproctitle
import torch
import wandb
from client.client_manager import ClientMananger
from data_preprocessing.data_loader import partition_data, get_dataloader
from mpi4py import MPI
from server.server_manager import ServerMananger
from attack_f3ba import Attack_F3BA
from attack_cerp import Attack_CERP
from attack_a3fl import Attack_A3FL
from attack_f3ba_resnet import Attack_F3BA_resnet
from attack_constrain import Attack_CONSTRAIN

from synthesizers.pattern_synthesizer import PatternSynthesizer as Syn
import yaml

# https://nyu-cds.github.io/python-mpi/05-collectives/
from model.FedNASAggregator import FedNASAggregator
from model.FedNASTrainer import FedNASTrainer
from tasks.fl.cifarfed_task import CifarFedTask #

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--stage', type=str, default='search',
                        help='stage: search; train')
    parser.add_argument('--model', type=str, default='resnet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--partition', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')
    parser.add_argument('--adv_epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally to optimize adv model')
    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')
    parser.add_argument('--client_number', type=int, default=4, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--comm_round', type=int, default=50,
                        help='how many round of communications we shoud use')

    parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='DARTS layers')

    parser.add_argument('--learning_rate', type=float, default=0.0025, help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')

    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--lambda_train_regularizer', type=float, default=1, help='train regularizer parameter')
    parser.add_argument('--lambda_valid_regularizer', type=float, default=1, help='validation regularizer parameter')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')

    parser.add_argument('--tau_max', type=float, default=10, help='initial tau')
    parser.add_argument('--tau_min', type=float, default=1, help='minimum tau')

    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--arch', type=str, default='FedNAS_V1', help='which architecture to use')
    parser.add_argument('--params', dest='params', default='configs/cifar_fed.yaml')
    parser.add_argument('--frequency_of_the_test', type=int, default=1, help='the frequency of the test')
    # parser.add_argument('--defense', type=bool, default=False, help='use defense')
    parser.add_argument('--defense_method', type=str, default='ESFL', help='use defense: SSD, CRFL, FEDRAD, ESFL')
    parser.add_argument('--sigma', type=float, default=0.025, help='sigma')
    parser.add_argument('--current_round', type=int, default=0, help='current round')
    parser.add_argument('--attack_type', type=str, default='a3fl', help='constrain, dba, f3ba, cerp, a3fl, arch')
    parser.add_argument('--participant_number', type=int, default=8, metavar='NN',
                        help='number of participants in each communication round')
    parser.add_argument('--attack_round', type=int, default=0, help='the round start to attack')  # start to attack
    parser.add_argument('--number_of_adversaries', type=int, default=2, help='number of adversaries')  # start to attack
    parser.add_argument('--search_space', type=str, default='GENOSPACE', help='type of search space: DARTS, BESTACC, GENOSPACE, TINYGENO')
    parser.add_argument('--model_arch', type=str, default='NetV1', help='NetV1, NetV2')
    parser.add_argument('--start_gpu', type=int, default=0, help='start gpu count')
    parser.add_argument('--offline', type=str, default='no', help='wandb offline')
    parser.add_argument('--notes', type=str, default=None, help='wandb notes')
    parser.add_argument('--alpha', type=int, default=0.5, help='non-iid level')
    parser.add_argument('--path', type=str, default="/root/", help='root path')
    parser.add_argument('--attack_scale', type=float, default=3, help='scale factor of constrain-and-scale')
    parser.add_argument('--start_round', type=int, default=0, help='the round start to attack')
    parser.add_argument('--end_round', type=int, default=500, help='the round ending attack')
    parser.add_argument('--portion', type=float, default=1, help='')
    parser.add_argument('--tags', type=str, default="", help='')
    parser.add_argument('--poison_type', type=str, default='MGDA', help='poison type')
    # parser.add_argument('--n_attackers', type=int, default=1, help='how many attackers')
    args = parser.parse_args()
    if args.defense_method == "None":
        args.defense_method = None
    return args

# newly added attackers represents attacker ids
def init_server(args, comm, rank, size, round_num, attackers, attack, attacker):
    # machine learning experiment tracking platform: https://www.wandb.com/
    tags = args.tags.split(",") if args.tags else []
    wandb.login(key = "c5de5a0aa7227b82f916269ddd52c0259e02569f")
    wandb.init(
        project="backdoor_federated_nas",
        name="FedNAS(d)" + str(args.partition) + "r" + str(args.comm_round) +
             "-e" +  str(args.epochs) +
             "-lr" + str(args.learning_rate) +
             "-st" + str(args.stage) +
             "-at" + str(args.attack_type) +
             "-an" + str(args.number_of_adversaries) +
             "-sw" + str(args.attack_scale) +
             "-c"  + str(args.portion) +
             "-df" + str(args.defense_method) +
             "-ds" + str(args.dataset) +
             "-cn" + str(args.client_number),
        config=args,
        notes=str(args.notes),
        tags=tags,
    )

    # load data
    logging.info("load dataset")
    if args.dataset == 'cifar10':
        args_datadir = os.path.join(args.path, "cifar10")
        args_logdir = os.path.join(args.path, "log/cifar10")
    elif args.dataset == 'tiny':
        args_datadir = os.path.join(args.path, "tiny")
        args_logdir = os.path.join(args.path, "log/tiny")

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset,
                                                                                             args_datadir,
                                                                                             args_logdir,
                                                                                             args.partition,
                                                                                             args.client_number,
                                                                                             args.alpha,
                                                                                             args=args)
    # FEDRAD
    _, _, _, _, unlabeled_dataidx_map, unlabeled_traindata_cls_counts = partition_data(args.dataset,
                                                                                       args_datadir,
                                                                                       args_logdir,
                                                                                       'homo',
                                                                                       args.client_number,
                                                                                       args.alpha,
                                                                                       args=args)

    n_classes = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    logging.info("net_dataidx_map = " + str(net_dataidx_map))
    logging.info("#####################")

    all_train_data_num = sum([len(unlabeled_dataidx_map[r]) for r in range(args.client_number)])
    dataidxs = unlabeled_dataidx_map[0]
    local_sample_number = len(dataidxs)
    split = int(np.floor(0.5 * local_sample_number))  # split index
    train_idxs = dataidxs[0:split]
    test_idxs = dataidxs[split:local_sample_number]

    # all_train_data_num = sum([len(net_dataidx_map[r]) for r in range(args.client_number)])
    _, test_global = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size)
    train_global, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size, train_idxs)

    logging.info("train_dl_global number = " + str(len(train_global)))
    logging.info("test_dl_global number = " + str(len(test_global)))

    # aggregator
    # client_num = size - 1
    client_num = args.client_number
    aggregator = FedNASAggregator(train_global, test_global, all_train_data_num, client_num, device, args, attack, attacker)

    # start the distributed training
    # server_manager = ServerMananger(args, comm, rank, size, round_num, aggregator)
    server_manager = ServerMananger(args, comm, rank, size, round_num, aggregator,args.number_of_adversaries,args.client_number)
    server_manager.run()


def init_client(args, params, comm, rank, size, round_num, seed, attack, attacker):
    # to make sure each client has the same initial weight
    torch.manual_seed(seed)
    attack = attack
    # client_ID = rank - 1
    client_ID = (rank - 1) % args.client_number

    # 1. load data
    logging.info("load dataset")
    if args.dataset == 'cifar10':
        args_datadir = os.path.join(args.path, "cifar10")
        args_logdir = os.path.join(args.path, "log/cifar10")
    elif args.dataset == 'tiny':
        args_datadir = os.path.join(args.path, "tiny")
        args_logdir = os.path.join(args.path, "log/tiny")

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args.dataset,
                                                                                             args_datadir,
                                                                                             args_logdir,
                                                                                             args.partition,
                                                                                             args.client_number,
                                                                                             args.alpha,
                                                                                             args=args)

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))

    _, _, _, _, unlabeled_dataidx_map, unlabeled_traindata_cls_counts = partition_data(args.dataset,
                                                                                       args_datadir,
                                                                                       args_logdir,
                                                                                       'homo',
                                                                                       args.client_number,
                                                                                       args.alpha,
                                                                                       args=args)
    unlabeled_all_train_data_num = sum([len(unlabeled_dataidx_map[r]) for r in range(args.client_number)])
    unlabeled_dataidxs = unlabeled_dataidx_map[client_ID]
    unlabeled_local_sample_number = len(unlabeled_dataidxs)
    split = int(np.floor(0.5 * unlabeled_local_sample_number))  # split index
    unlabeled_train_idxs = unlabeled_dataidxs[0:split]

    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    all_train_data_num = sum([len(net_dataidx_map[r]) for r in range(args.client_number)])
    dataidxs = net_dataidx_map[client_ID]
    # logging.info("rank = %d, dataidxs = %s" % (rank, dataidxs))
    local_sample_number = len(dataidxs)
    logging.info("rank = %d, local_sample_number = %d" % (rank, local_sample_number))

    split = int(np.floor(0.5 * local_sample_number))  # split index
    train_idxs = dataidxs[0:split]
    test_idxs = dataidxs[split:local_sample_number]

    train_local, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size, train_idxs)
    logging.info("rank = %d, batch_num_train_local = %d" % (rank, len(train_local)))

    test_local, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size, test_idxs)
    logging.info("rank = %d, batch_num_test_local = %d" % (rank, len(test_local)))
    if args.defense_method == 'FEDRAD':
        unlabeled_train_local, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size,
                                                  unlabeled_train_idxs)
        # unlabeled_train_local, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size)

        trainer = FedNASTrainer(client_ID, train_local, test_local, unlabeled_train_local, local_sample_number,
                                all_train_data_num, device, args, params, attack, attacker)
    else:
        trainer = FedNASTrainer(client_ID, train_local, test_local, None, local_sample_number,
                                all_train_data_num, device, args, params, attack, attacker)

    # 3. start the distributed training
    client_manager = ClientMananger(args, comm, rank, size, round_num, trainer)
    client_manager.run()


def init_training_device(process_ID, size):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    # if process_ID == 0 or process_ID == 1:
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     return device
    # process_gpu_dict = dict()
    # gpu_number = args.gpu
    #
    # for client_index in range(size - 1):
    #     gpu_index = client_index % (gpu_number - 1)
    #     process_gpu_dict[client_index] = gpu_index + 1
        
    # if process_ID == 0:
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     return device
    # process_gpu_dict = dict()
    # gpu_number = args.gpu
    # for client_index in range(size - 1):
    #       gpu_index = client_index % gpu_number
    #       process_gpu_dict[client_index] = gpu_index
    #
    # logging.info(process_gpu_dict)
    # device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    # logging.info(device)

    gpu_number = args.gpu
    k = args.start_gpu
    if process_ID == 0:
        device = torch.device(f"cuda:{k}" if torch.cuda.is_available() else "cpu")
        return device

    process_gpu_dict = dict()

    for client_index in range(size - 1):
        gpu_index = (client_index % gpu_number) + k
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = add_args(parser)
    if args.offline == 'yes':
        wandb.init(mode="offline")

    str_process_name = "Federated Learning:" + str(rank)
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params = Params(**params)
    if args.dataset == 'cifar10':
        params.task = 'CIFAR'
    elif args.dataset == 'tiny':
        params.task = 'TINY'
    params.handcraft = False
    params.handcraft_trigger = False
    if args.partition == 'hetero':
        params.heterogenuity = args.alpha
        # params.heterogenuity = 1.0
    else:
        params.heterogenuity = 1.0
    params.fl_number_of_adversaries = args.number_of_adversaries
    params.data_path = args.path
    params.poisoning_proportion = args.portion
    setproctitle.setproctitle(str_process_name)

    logging.basicConfig(level=logging.INFO,
                        format=str(rank) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(rank) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid()))
                 +", The number of attackers:"+str(params.fl_number_of_adversaries))

    # Set the random seed if provided (affects client sampling, and batching)
    # if a pseudorandom number generator is reinitialized with the same seed
    # it will produce the same sequence of numbers.
    seed = 0
    np.random.seed(seed)

    client_seed = np.random.randint(size)

    logging.info("rank = %d, size = %d" % (rank, size))
    device = init_training_device(rank, size)
    Params.device = device
    Params.fl_weight_scale = args.attack_scale
    round_num = args.comm_round
    attackers = []
    if args.number_of_adversaries > 0:
        for i in range(0, args.number_of_adversaries):
            attackers.append(i+1)
    logging.info("The number of attackers:"+str(args.number_of_adversaries))
    #   syn = Syn()
    #   attacker = Attack(params, syn, args)

    if rank == 0:
        syn = Syn((0, args.number_of_adversaries), args)

        if args.attack_type == 'f3ba' and args.stage != 'resnet':
            params.handcraft = True
            params.handcraft_trigger = True
            attacker = Attack_F3BA(params, syn, args)
        elif args.attack_type == 'cerp':
            attacker = Attack_CERP(params, syn, args)
        elif args.attack_type == 'a3fl':
            attacker = Attack_A3FL(params, syn, args)
        elif args.attack_type == 'f3ba' and args.stage == 'resnet':
            params.handcraft = True
            params.handcraft_trigger = True
            attacker = Attack_F3BA_resnet(params, syn)
        elif args.attack_type == 'constrain':
            attacker = Attack_CONSTRAIN(params, syn)
        elif args.attack_type == 'arch':
            attacker = Attack_CONSTRAIN(params, syn)
        else:
            attacker = Attack_CONSTRAIN(params, syn)
        # init_server(args, comm, rank, size, round_num, attack=True, attacker=attacker)
        init_server(args, comm, rank, size, round_num, attackers, attack=True, attacker=attacker)
    else:
        syn = Syn((rank, args.number_of_adversaries), args)

        if args.attack_type == 'f3ba' and args.stage != 'resnet':
            params.handcraft = True
            params.handcraft_trigger = True
            attacker = Attack_F3BA(params, syn, args)
        elif args.attack_type == 'cerp':
            attacker = Attack_CERP(params, syn, args)
        elif args.attack_type == 'a3fl':
            attacker = Attack_A3FL(params, syn, args)
        elif args.attack_type == 'f3ba' and args.stage == 'resnet':
            params.handcraft = True
            params.handcraft_trigger = True
            attacker = Attack_F3BA_resnet(params, syn)
        elif args.attack_type == 'constrain':
            attacker = Attack_CONSTRAIN(params, syn)
        elif args.attack_type == 'arch':
            attacker = Attack_CONSTRAIN(params, syn)
        else:
            attacker = Attack_CONSTRAIN(params, syn)
        if rank in attackers:

            logging.info("===========================Backdoor on==========================================")
            init_client(args, params, comm, rank, size, round_num, client_seed, attack=True, attacker=attacker)
        else:

            init_client(args, params, comm, rank, size, round_num, client_seed, attack=False, attacker=attacker)
