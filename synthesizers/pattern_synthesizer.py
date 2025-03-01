import random
from tasks.batch import Batch
import torch
from torchvision.transforms import transforms, functional
from utils.parameters import Params
from tasks.task import Task

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class PatternSynthesizer(object):
#    pattern_tensor: torch.Tensor = torch.tensor([
#        [1., 0., 1.],
#        [-10., 1., -10.],
#        [-10., -10., 0.],
#        [-10., 1., -10.],
#        [1., 0., 1.]
#    ])
#

    params = Params(loss_tasks=['normal', 'backdoor'])
    
    def __init__(self, mal: tuple, args):
        self.pattern_tensor: torch.Tensor = torch.tensor([
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ])

        "Just some random 2D pattern."

        self.x_top = 3
        "X coordinate to put the backdoor into."
        self.y_top = 23
        "Y coordinate to put the backdoor into."

        self.x_bot = self.x_top + self.pattern_tensor.shape[0]
        self.y_bot = self.y_top + self.pattern_tensor.shape[1]
        dbas = []

        self.mask_value = -10
        "A tensor coordinate with this value won't be applied to the image."

        self.resize_scale = (5, 10)
        "If the pattern is dynamically placed, resize the pattern."

        self.mask: torch.Tensor = None
        "A mask used to combine backdoor pattern with the original image."

        self.pattern: torch.Tensor = None
        "A tensor of the `input.shape` filled with `mask_value` except backdoor."
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2023, 0.1994, 0.2010))
        if args.dataset == 'cifar10':
            self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
        elif args.dataset == 'tiny':
            self.normalize = transforms.Normalize((0.4802, 0.4481, 0.3975),
                                             (0.2302, 0.2265, 0.2262))



        self.i_mal = mal[0]
        self.n_mal = mal[1]

        self.make_pattern(self.pattern_tensor, self.x_top, self.y_top)
        self.initial_tensor = self.pattern
        self.params = Params(loss_tasks=['normal', 'backdoor'])
        if self.i_mal != 0 and self.i_mal <= self.n_mal:
            if args.attack_type != "constrain":
                self.random_break_trigger()

    def make_pattern(self, pattern_tensor, x_top, y_top):
        trigger_size = (3, 3)
        if self.params.handcraft:
            torch.manual_seed(111)
            pattern_tensor = torch.rand(trigger_size)
            #   print('Initial Tensor:\n', pattern_tensor)
            pattern_tensor = (pattern_tensor * 255).floor() / 255
        else:
            pattern_tensor = torch.zeros(trigger_size)
        self.x_bot = self.x_top + pattern_tensor.shape[0]
        self.y_bot = self.y_top + pattern_tensor.shape[1]
        full_image = torch.zeros(self.params.input_shape).fill_(self.mask_value)
        
        x_bot = self.x_bot
        y_bot = self.y_bot              
        #   full_image = torch.zeros(self.params.input_shape)
        #   full_image.fill_(self.mask_value)
        #   x_bot = x_top + pattern_tensor.shape[0]
        #    y_bot = y_top + pattern_tensor.shape[1]
        if x_bot >= self.params.input_shape[1] or \
                y_bot >= self.params.input_shape[2]:
            raise ValueError(f'Position of backdoor outside image limits:'
                             f'image: {self.params.input_shape}, but backdoor'
                             f'ends at ({x_bot}, {y_bot})')

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)
        self.pattern = self.normalize(full_image).to(self.params.device)

    # if number of attackers > 1
    def set_imal(self, imal):
        self.i_mal = imal

    def random_break_trigger(self):
        x_top, y_top = self.x_top, self.y_top
        i_mal, n_mal = self.i_mal, self.n_mal
        assert (n_mal in [1, 2, 4])
        if n_mal == 1:
            for p in range(3):
                gx = random.randint(0, 2)
                gy = random.randint(0, 2)
                self.mask[:,x_top + gx, y_top + gy] = 0
        elif n_mal == 2:
            if i_mal == 1:
                self.mask[:,x_top, y_top] = 0
                self.mask[:,x_top + 2, y_top] = 0
                self.mask[:,x_top + 2, y_top] = 0
                self.mask[:,x_top + 2, y_top + 2] = 0
            elif i_mal == 2:
                self.mask[:,x_top, y_top + 1] = 0
                self.mask[:,x_top + 2, y_top + 1] = 0
                self.mask[:,x_top + 1, y_top] = 0
                self.mask[:,x_top + 1, y_top + 2] = 0
            else:
                raise ValueError("out of mal index!")
            print("dba mask:{}:\n".format((i_mal,n_mal)),self.mask[0, 3:7, 23:27])
        elif n_mal == 4:
            if i_mal == 1:
                self.mask[:,x_top, y_top] = 0
                self.mask[:,x_top + 1, y_top] = 0
                self.mask[:,x_top, y_top + 1] = 0
            if i_mal == 2:
                self.mask[:,x_top, y_top + 2] = 0
                self.mask[:,x_top + 1, y_top + 2] = 0
                self.mask[:,x_top, y_top + 1] = 0
            if i_mal == 3:
                self.mask[:,x_top + 2, y_top] = 0
                self.mask[:,x_top + 2, y_top + 1] = 0
                self.mask[:,x_top + 1, y_top + 0] = 0
            if i_mal == 4:
                self.mask[:,x_top + 2, y_top + 2] = 0
                self.mask[:,x_top + 1, y_top + 2] = 0
                self.mask[:,x_top + 2, y_top + 1] = 0
            print("dba mask:{}:\n".format((i_mal,n_mal)),self.mask[0,x_top:x_top+4, y_top:y_top+4])
        else:
            raise ValueError("Not implement DBA for num of clients out of 1,2,4")



    def apply_backdoor(self, batch, attack_portion):
        self.synthesize_inputs(batch=batch, attack_portion=attack_portion)
        self.synthesize_labels(batch=batch, attack_portion=attack_portion)
        return

    def make_backdoor_batch(self, batch: Batch, test=False, attack=True) -> Batch:
        # syn = PatternSynthesizer()

        # Don't attack if only normal loss task.
        if (not attack) or (self.params.loss_tasks == ['normal'] and not test):
            return batch

        if test:
            attack_portion = batch.batch_size
        else:
            attack_portion = round(
                batch.batch_size * self.params.poisoning_proportion)

        backdoored_batch = batch.clone()
        self.apply_backdoor(batch=backdoored_batch, attack_portion=attack_portion)
        return backdoored_batch

    def synthesize_inputs(self, batch, attack_portion=None):
        pattern, mask = self.get_pattern()
        device = "cuda:"+str(mask.get_device())
        _device = "cuda:"+str(batch.inputs.get_device())
        pattern = pattern.to(device)
        batch.inputs = batch.inputs.to(device)
        batch.inputs[:attack_portion] = (1 - mask) * \
                                        batch.inputs[:attack_portion] + \
                                        mask * pattern
        batch.inputs = batch.inputs = batch.inputs.to(_device)
        return

    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)
        return



    def get_pattern(self):
        if self.params.backdoor_dynamic_position:
            resize = random.randint(self.resize_scale[0], self.resize_scale[1])
            pattern = self.pattern_tensor
            if random.random() > 0.5:
                pattern = functional.hflip(pattern)
            image = transform_to_image(pattern)
            pattern = transform_to_tensor(
                functional.resize(image,
                resize, interpolation=0)).squeeze()

            x = random.randint(0, self.params.input_shape[1] \
                               - pattern.shape[0] - 1)
            y = random.randint(0, self.params.input_shape[2] \
                               - pattern.shape[1] - 1)
            self.make_pattern(pattern, x, y)

        return self.pattern, self.mask

