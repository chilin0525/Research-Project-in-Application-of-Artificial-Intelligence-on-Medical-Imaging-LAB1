from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np


class DataAugmentation():
    def __init__(self, img_size):
        self.img_size = img_size

    def transform(self, img):
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=0)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img


class Dataset():
    def __init__(
        self, 
        files,
        label_list,
        img_size=256, 
        is_train=True,
        to_tensor=True
    ):
        self.img_size = img_size
        self.files = files
        self.label_list = label_list
        self.A_size = len(self.files)
        if is_train:
            self.augm = DataAugmentation(
                img_size = self.img_size
            )
        else:
            self.augm = DataAugmentation(
                img_size = self.img_size
            )

    def __getitem__(
        self, 
        index
    ):
        file_path = self.files[index]
        img = Image.open(file_path).convert('RGB')
        img, gt = self.augm.transform(img), self.label_list[index]
        return img, gt

    def __len__(self):
        return len(self.files)


def get_loader(
    files,
    gt,
    shuffle: bool = False,
    img_size: int = 256, 
    batch_size: int = 8, 
    is_train: bool = False,
    num_workers: int = 4,
):
    dataset = Dataset(
        files,
        gt,
        img_size = img_size, 
        is_train = is_train,
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle, 
        num_workers=4
    )
    return dataloader