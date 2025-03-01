import logging
import sys
import time

from communication.com_manager import CommunicationManager
from communication.mpi_message import MPIMessage
from communication.observer import Observer


class Client(object):

    def __init__(self, args, comm, rank, size, round_num, trainer):
        self.args = args
        self.size = size
        self.rank = rank
        self.trainer = trainer
        self.num_rounds = round_num
        self.round_idx = 0


    def receive_config(self, global_model_params, arch_params):
        self.trainer.update_model(global_model_params)
        if self.args.stage == "search":
            self.trainer.update_arch(arch_params)

        self.round_idx = 0
        # start to train
        self.send_model()

    def receive_model_from_server(self, model_params, arch_params):

        self.trainer.update_model(model_params)
        if self.args.stage == "search":
            self.trainer.update_arch(arch_params)

        self.round_idx += 1
        self.send_model()

    def send_model(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        start_time = time.time()
        if self.args.stage == "search":
            weights, alphas, local_sample_num, train_acc, train_loss = self.trainer.search()
        else:
            weights, local_sample_num, train_acc, train_loss = self.trainer.train()
            alphas = []
        train_finished_time = time.time()
        # for one epoch, the local searching time cost is: 75s (based on RTX2080Ti)
        logging.info("local searching time cost: %d" % (train_finished_time - start_time))

        """
        In order to maintain the flexibility and track the asynchronous property, we temporarily don't use "comm.reduce".
        The communication speed of CUDA-aware MPI is faster than regular MPI. 
        According to this document:
        http://on-demand.gputechconf.com/gtc/2014/presentations/S4236-multi-gpu-programming-mpi.pdf,
        when the message size is around 4 Megabytes, CUDA-aware MPI is three times faster than regular MPI.
        In our case, the model size of ResNet is around 229.62M. Thus, we will optimize the communication speed using
        CUDA-aware MPI.
        """
        return weights, alphas, local_sample_num, train_acc, train_loss




