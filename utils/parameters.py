from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import List, Dict
import logging
import torch
logger = logging.getLogger('logger')

ALL_TASKS =  ['backdoor', 'normal', 'sentinet_evasion', #'spectral_evasion',
                           'neural_cleanse', 'mask_norm', 'sums', 'neural_cleanse_part1']

@dataclass
class Params:

    # Corresponds to the class module: tasks.mnist_task.MNISTTask
    # See other tasks in the task folder.
    task: str = 'CIFAR'

    current_time: str = None
    name: str = None
    commit: float = None
    random_seed: int = None
    device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # training params
    start_epoch: int = 1
    epochs: int = None
    log_interval: int = 1000

    # model arch is usually defined by the task
    pretrained: bool = False
    resume_model: str = None
    lr: float = None
    decay: float = None
    momentum: float = None
    optimizer: str = None
    scheduler: bool = False
    scheduler_milestones: List[int] = None
    # data
    data_path: str = '/root/'
    batch_size: int = 128
    test_batch_size: int = 100
    transform_train: bool = True
    "Do not apply transformations to the training images."
    max_batch_id: int = None
    "For large datasets stop training earlier."
    input_shape = torch.Size([3, 32, 32])
    "No need to set, updated by the Task class."

    # gradient shaping/DP params
    dp: bool = None
    dp_clip: float = None
    dp_sigma: float = None

    # attack params
    backdoor: bool = False
    backdoor_label: int = 8
    poisoning_proportion: float = 1  # backdoors proportion in backdoor loss
    synthesizer: str = 'pattern'
    backdoor_dynamic_position: bool = False

    # nc evasion
    nc_p_norm: int = 1

    # losses to balance: `normal`, `backdoor`, `neural_cleanse`, `sentinet`,
    # `backdoor_multi`.
    #loss_tasks: List[str] = None
    loss_tasks: List = field(default_factory=lambda: ['noraml', 'backdoor']) # F3BA
    loss_balance: str = 'MGDA'
    "loss_balancing: `fixed` or `MGDA`"

    loss_threshold: float = None

    # approaches to balance losses with MGDA: `none`, `loss`,
    # `loss+`, `l2`
    mgda_normalize: str = None
    fixed_scales: Dict[str, float] = None
    # fixed_scales: Dict[str, float] = field(default_factory=lambda: {'normal':0.6,'backdoor':0.4}) # F3BA
    # relabel images with poison_number
    poison_images: List[int] = None
    poison_images_test: List[int] = None
    # optimizations:
    alternating_attack: float = None
    clip_batch: float = None
    # Disable BatchNorm and Dropout
    switch_to_eval: float = None

    # nc evasion
    nc_p_norm: int = 1
    # spectral evasion
    spectral_similarity: 'str' = 'norm'

    # handcrafted trigger(F3BA)
    handcraft = False
    handcraft_trigger = False
    distributed_trigger = False
    # F3BA flip rate $$$$$
    conv_rate: float = 0.02
    fc_rate: float = 0.001
    fixed_scales: Dict[str, float] = field(default_factory=lambda: {'normal': 0.6, 'backdoor': 0.4})
    # fixed_mal =True

    acc_threshold = 0.01 # F3BA
    # For flip(F3BA)
    flip_factor = 1
    alpha_loss: float = 0.0001
    # if attack bulyan, should set a number>0, for example 0.7（F3BA）
    model_similarity_factor: float = 0.0
    # freeze
    freezing = False
    norm_clip_factor: float = 10.0

    heterogenuity: float = 1.0  # unkown
    # differential privacy（F3BA）
    dp: bool = False
    kernel_selection: str = "movement"  # F3BA

    # server_dataset（F3BA）
    server_dataset = False
    resultdir = 'result-fedavg'

    # logging
    report_train_loss: bool = True
    log: bool = False
    tb: bool = False
    save_model: bool = None
    save_on_epochs: List[int] = None
    save_scale_values: bool = False
    print_memory_consumption: bool = False
    save_timing: bool = False
    timing_data = None

    # Temporary storage for running values
    running_losses = None
    running_scales = None

    # FL params
    fl: bool = True
    fl_no_models: int = 100
    fl_local_epochs: int = 2
    fl_total_participants: int = 20
    fl_eta: int = 1
    fl_sample_dirichlet: bool = False
    fl_dirichlet_alpha: float = None
    fl_diff_privacy: bool = False
    fl_dp_clip: float = None
    fl_dp_noise: float = None
    # FL attack details. Set no adversaries to perform the attack:
    fl_number_of_adversaries: int = 2
    fl_single_epoch_attack: int = None
    fl_weight_scale: int = 3

    def __post_init__(self):
        # enable logging anyways when saving statistics
        if self.save_model or self.tb or self.save_timing or \
                self.print_memory_consumption:
            self.log = True

        if self.log:
            self.folder_path = f'saved_models/model_' \
                               f'{self.task}_{self.current_time}_{self.name}'

        self.running_losses = defaultdict(list)
        self.running_scales = defaultdict(list)
        self.timing_data = defaultdict(list)

        for t in self.loss_tasks:
            if t not in ALL_TASKS:
                raise ValueError(f'Task {t} is not part of the supported '
                                 f'tasks: {ALL_TASKS}.')

    def to_dict(self):
        return asdict(self)