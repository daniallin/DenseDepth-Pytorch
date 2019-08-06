import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Compose, ToPILImage, ToTensor
import torch.utils.data as data

from dataloader.preprocessing import RandomHorizontalFlip, RandomCropRotate


class NYU2Dataset(data.Dataset):
    def __init__(self, path='data/', train=True):
        self.train = train
        if train:
            train_data = pd.read_csv(path+'nyu2_train.csv', header=None)

            self.img_names = train_data.iloc[:, 0]
            self.depth_names = train_data.iloc[:, 1]
        else:
            val_data = pd.read_csv(path+'nyu2_test.csv', header=None)

            self.img_names = val_data.iloc[:, 0]
            self.depth_names = val_data.iloc[:, 1]

        self.train_flip = Compose([RandomHorizontalFlip(), RandomCropRotate(angle=45)])
        self.general_transform = Compose([ToPILImage(), ToTensor()])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name, depth_name = self.img_names[index], self.depth_names[index]
        img = Image.open(img_name)
        depth = Image.open(depth_name)

        if self.train:
            img, depth = np.array(img), np.array(depth)[:, :, None]
            img, depth = self.train_flip((img, depth))

        img, depth = self.general_transform((img, depth))

        return img, depth

