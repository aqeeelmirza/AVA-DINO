import os
import PIL
import torch
from torchvision import transforms
import numpy as np
import tifffile as tiff
from enum import Enum

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TEST = "test"

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TEST,
        transform_img=None,
        transform_mask=None,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split

        self.image_dir = os.path.join(root_dir, "test-image")
        self.mask_dir = os.path.join(root_dir, "test-mask")

        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        assert len(self.image_files) == len(self.mask_files), "len(image_files) != len(mask_files)"

        if transform_img is None:
            self.transform_img = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform_img = transform_img

        if transform_mask is None:
            self.transform_mask = transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ])
        else:
            self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = PIL.Image.open(image_path).convert("RGB")
        mask = PIL.Image.open(mask_path).convert("L")

        image = self.transform_img(image)
        mask = self.transform_mask(mask)

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": 1,
            "image_path": image_path,
        }
