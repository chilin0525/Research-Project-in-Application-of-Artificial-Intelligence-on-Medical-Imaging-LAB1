from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import v2
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch 


class DataAugmentation():
    def __init__(self, 
                 img_size, 
                 is_RandomRotation = False,
                 is_RandomHorizontalFlip = False,
                 is_RandomResizedCrop = False,
                 is_GaussianBlur = False,
                 is_RandomPerspective = False,
                 is_CenterCrop = False,
    ):

        self.img_size = img_size
        self.is_RandomResizedCrop = is_RandomResizedCrop
        self.is_RandomHorizontalFlip = is_RandomHorizontalFlip
        self.is_RandomRotation = is_RandomRotation
        self.is_GaussianBlur = is_GaussianBlur
        self.is_RandomPerspective = is_RandomPerspective
        self.is_CenterCrop = is_CenterCrop

        aug_method_list = []        

        if self.is_RandomHorizontalFlip:
            aug_method_list.append(v2.RandomHorizontalFlip(p=0.5))
        
        if self.is_RandomRotation:
            aug_method_list.append(v2.RandomRotation(degrees=(-15, 15)))

        if self.is_GaussianBlur:
            aug_method_list.append(v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)))

        if self.is_RandomPerspective:
            aug_method_list.append( v2.RandomPerspective(distortion_scale=0.6, p=1.0) )

        if self.is_CenterCrop:
            aug_method_list.append( v2.CenterCrop(int(self.img_size*0.9)) )

        aug_method_list.append( transforms.Resize((self.img_size, self.img_size)) )
        aug_method_list.append( transforms.ToTensor() )
        aug_method_list.append( v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transforms = transforms.Compose(aug_method_list)
    
    def transform(self, img):
        return self.transforms(img)


class Dataset():
    def __init__(
        self, 
        files,
        label_list,
        img_size = 256, 
        is_train = True,
        to_tensor = True,
        is_RandomRotation = False,
        is_RandomHorizontalFlip = False,
        is_GaussianBlur = False,
        is_RandomPerspective = False,
        is_CenterCrop = False,
    ):
        self.img_size = img_size
        self.files = files
        self.label_list = label_list
        self.A_size = len(self.files)

        self.is_RandomRotation = is_RandomRotation
        self.is_RandomHorizontalFlip = is_RandomHorizontalFlip
        self.is_GaussianBlur = is_GaussianBlur
        
        if is_train:
            self.augm = DataAugmentation(
                img_size = self.img_size,
                is_RandomRotation = is_RandomRotation,
                is_RandomHorizontalFlip = is_RandomHorizontalFlip,
                is_GaussianBlur = is_GaussianBlur,
                is_RandomPerspective = is_RandomPerspective,
                is_CenterCrop = is_CenterCrop
            )
        else:
            # inference, expect do not open aug
            self.augm = DataAugmentation(
                img_size = self.img_size,
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
    is_RandomRotation: bool = False,
    is_RandomHorizontalFlip: bool = False,
    is_GaussianBlur: bool = False,
    is_RandomPerspective: bool = False,
    is_CenterCrop: bool = False,
):
    dataset = Dataset(
        files,
        gt,
        img_size = img_size, 
        is_train = is_train,
        is_RandomRotation = is_RandomRotation,
        is_RandomHorizontalFlip = is_RandomHorizontalFlip,
        is_GaussianBlur = is_GaussianBlur,
        is_RandomPerspective = is_RandomPerspective,
        is_CenterCrop = is_CenterCrop
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle, 
        num_workers = num_workers,
        pin_memory = True,
    )
    return dataloader