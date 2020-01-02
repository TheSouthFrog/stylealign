import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, get_list
import random
import torch
import numpy as np

class UnalignedLandmarkDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_A = os.path.join(opt.dir_landmarks_list_A, opt.name_landmarks_list)
        self.dir_B = os.path.join(opt.dir_landmarks_list_B, opt.name_landmarks_list)

        self.A_items = get_list(self.dir_A)
        self.B_items = get_list(self.dir_B)

        self.A_size = len(self.A_items)
        self.B_size = len(self.B_items)
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_label = self.A_items[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_label = self.B_items[index_B]

        A_label = np.asarray(A_label)
        A_label = torch.from_numpy(A_label).float()

        B_label = np.asarray(B_label)
        B_label = torch.from_numpy(B_label).float()
        # A = self.transform(A_label)
        # B = self.transform(B_label)

        return {'A': A_label, 'B': B_label,
                'A_paths': self.dir_A, 'B_paths': self.dir_B}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedLandmarkDataset'
