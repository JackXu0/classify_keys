import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import os
import numpy as np


class myDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(self.root_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        key_type = self.files[idx].split("_")[0]

        image = np.array(Image.open(os.path.join(self.root_dir + self.files[idx])))

        image = torchvision.transforms.Resize((384, 216), interpolation=Image.NEAREST)(Image.fromarray(image))

        image = torchvision.transforms.ToTensor()(image)

        return image, torch.tensor(int(key_type))


class myPredictionDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = [f for f in os.listdir(self.root_dir) if not f.startswith('.')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        image = np.array(Image.open(os.path.join(self.root_dir + self.files[idx])))

        image = torchvision.transforms.Resize((384, 216), interpolation=Image.NEAREST)(Image.fromarray(image))

        image = torchvision.transforms.ToTensor()(image)

        return image