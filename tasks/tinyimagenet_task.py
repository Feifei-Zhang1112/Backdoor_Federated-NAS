import os
from models.resnet import resnet18
import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms
import torch.nn as nn
from models.simple import SimpleNet
from tasks.task import Task
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import shutil
class TinyImageNetTask(Task):

    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975),
                                     (0.2302, 0.2265, 0.2262))

    def load_data(self):
        self.load_tiny_imagenet_data()
    def load_tiny_imagenet_data(self):

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(36),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            self.normalize,
        ])
        num_classes = 80
        data_dir = self.params.data_path
        train_dir = os.path.join(data_dir, 'tiny/train')
        all_wnids = sorted(os.listdir(train_dir))
        selected_wnids = set(all_wnids[:num_classes])
        self.classes = all_wnids[:num_classes]
        train_dataset_all = datasets.ImageFolder(root=os.path.join(data_dir, 'tiny/train'), transform=train_transform)
        test_dataset_all = datasets.ImageFolder(root=os.path.join(data_dir, 'tiny/val'), transform=test_transform)


        train_dataset_all.samples = [(img_path, label) for img_path, label in train_dataset_all.samples if
                                     train_dataset_all.classes[label] in selected_wnids]
        train_dataset_all.imgs = train_dataset_all.samples

        test_dataset_all.samples = [(img_path, label) for img_path, label in test_dataset_all.samples if
                                    test_dataset_all.classes[label] in selected_wnids]
        test_dataset_all.imgs = test_dataset_all.samples


        train_dataset_all.classes = list(selected_wnids)
        train_dataset_all.class_to_idx = {cls: idx for idx, cls in enumerate(train_dataset_all.classes)}

        test_dataset_all.classes = list(selected_wnids)
        test_dataset_all.class_to_idx = {cls: idx for idx, cls in enumerate(test_dataset_all.classes)}

        self.train_dataset = train_dataset_all
        self.test_dataset = test_dataset_all
        self.train_loader = DataLoader(train_dataset_all, batch_size=self.params.batch_size, shuffle=True,
                                       num_workers=8, pin_memory=True)
        self.test_loader = DataLoader(test_dataset_all, batch_size=self.params.test_batch_size, shuffle=False,
                                      num_workers=8, pin_memory=True)


        # self.classes = [d.name for d in os.scandir(train_dir) if d.is_dir()]

    def build_model(self) -> nn.Module:
        model = resnet18(pretrained=self.params.pretrained)
        model.fc = nn.Linear(model.fc.in_features, 200)
        return model
