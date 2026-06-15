"""
Kvasir-SEG Dataset Loader for Polyp Segmentation
"""

import os
from enum import Enum
import PIL
import torch
from torchvision import transforms
import random

_CLASSNAMES = ["01"]  # Kvasir-SEG has only one class (polyp detection)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        clip_transformer=None,
        k_shot=0,
        random_seed=42,
        divide_num=1,
        divide_iter=0,
        train_ratio=0.7,
        val_ratio=0.15,
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        
        if classname == 'ALL':
            self.classnames_to_use = _CLASSNAMES
        else:
            self.classnames_to_use = [classname]

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data(random_seed)
        
        if divide_num > 1:
            self.data_to_iterate = self.sub_datasets(self.data_to_iterate, divide_num, divide_iter, random_seed)
        
        if k_shot > 0:
            torch.manual_seed(random_seed)
            if k_shot >= len(self.data_to_iterate):
                pass
            else:
                indices = torch.randint(0, len(self.data_to_iterate), (k_shot,))
                self.data_to_iterate = [self.data_to_iterate[i] for i in indices]
        
        if clip_transformer is None:
            self.transform_img = [
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_img = transforms.Compose(self.transform_img)
        else:
            self.transform_img = clip_transformer

        self.transform_mask = [
            transforms.Resize((resize, resize)),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def sub_datasets(self, full_datasets, divide_num, divide_iter, random_seed=42):
        if divide_num == 0:
            return full_datasets
        random.seed(random_seed)
        
        id_dict = {}
        for i in range(len(full_datasets)):
            anomaly_type = full_datasets[i][1]  # "Normal" or "Anomaly"
            if anomaly_type not in id_dict.keys():
                id_dict[anomaly_type] = []
            id_dict[anomaly_type].append(i)
        
        sub_id_list = []
        for k in id_dict.keys():
            type_id_list = id_dict[k]
            random.shuffle(type_id_list)
            devide_list = [type_id_list[i:i+divide_num] for i in range(0, len(type_id_list), divide_num)]
            sub_list = [devide_list[i][divide_iter] for i in range(len(devide_list)) if len(devide_list[i])>divide_iter]
            sub_id_list.extend(sub_list)
        
        return [full_datasets[id] for id in sub_id_list]

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        # Load mask if available (all Kvasir-SEG images have polyps, so all have masks)
        if mask_path is not None and os.path.exists(mask_path):
            mask = PIL.Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask) > 0  # Binarize mask
        else:
            # For normal images (if any) or if mask doesn't exist
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": int(anomaly != "Normal"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self, random_seed=42):
        """
        Load Kvasir-SEG dataset
        All images contain polyps (anomalies) with corresponding masks
        """
        data_to_iterate = []
        
        images_dir = os.path.join(self.source, 'images')
        masks_dir = os.path.join(self.source, 'masks')
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        if not os.path.exists(masks_dir):
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
        
        # Get all images
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"Total Kvasir-SEG dataset: {len(image_files)} polyp images")
        
        # Create dataset - all images are anomalies (polyps) with masks
        all_data = []
        
        for classname in self.classnames_to_use:
            for img_file in image_files:
                image_path = os.path.join(images_dir, img_file)
                
                # Find corresponding mask (same filename in masks folder)
                mask_path = os.path.join(masks_dir, img_file)
                
                if not os.path.exists(mask_path):
                    print(f"⚠️  Warning: Mask not found for {img_file}")
                    mask_path = None
                
                # All images are anomalies (contain polyps)
                all_data.append((classname, "Anomaly", image_path, mask_path))
        
        # Split data into train/val/test
        random.seed(random_seed)
        random.shuffle(all_data)
        
        total = len(all_data)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        if self.split == DatasetSplit.TRAIN:
            data_to_iterate = all_data[:train_end]
        elif self.split == DatasetSplit.VAL:
            data_to_iterate = all_data[train_end:val_end]
        else:  # TEST
            data_to_iterate = all_data[val_end:]
        
        print(f"Kvasir-SEG {self.split.value} set: {len(data_to_iterate)} images (all polyps)")

        return None, data_to_iterate