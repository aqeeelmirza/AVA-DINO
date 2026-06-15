"""
MVTec AD-2 Dataset Loader
"""

import os
from enum import Enum
import PIL
import torch
from torchvision import transforms
import random

_CLASSNAMES = ["fabric", "can", "vial", "rice", "walnuts", 
               "fruit_jelly", "wallplugs", "sheet_metal"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test_public"

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
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        
        if classname == 'ALL':
            self.classnames_to_use = _CLASSNAMES
        else:
            self.classnames_to_use = [classname]

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        
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
            anomaly_type = full_datasets[i][1]  # "good" or "bad"
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

        if self.split == DatasetSplit.TEST and mask_path is not None and os.path.exists(mask_path):
            mask = PIL.Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask) > 0
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": int(anomaly == "bad"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        data_to_iterate = []
        
        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            
            if not os.path.exists(classpath):
                continue
            
            imgpaths_per_class[classname] = {}
            
            # For train and validation splits - only "good" folder
            if self.split in [DatasetSplit.TRAIN, DatasetSplit.VAL]:
                good_dir = os.path.join(classpath, "good")
                if os.path.exists(good_dir):
                    good_files = sorted([f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg'))])
                    imgpaths_per_class[classname]["good"] = [os.path.join(good_dir, f) for f in good_files]
                    
                    for image_path in imgpaths_per_class[classname]["good"]:
                        data_tuple = (classname, "good", image_path, None)
                        data_to_iterate.append(data_tuple)
            
            # For test split - "good" and "bad" folders
            elif self.split == DatasetSplit.TEST:
                # Good images
                good_dir = os.path.join(classpath, "good")
                if os.path.exists(good_dir):
                    good_files = sorted([f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg'))])
                    imgpaths_per_class[classname]["good"] = [os.path.join(good_dir, f) for f in good_files]
                    
                    for image_path in imgpaths_per_class[classname]["good"]:
                        data_tuple = (classname, "good", image_path, None)
                        data_to_iterate.append(data_tuple)
                
                # Bad images with masks
                bad_dir = os.path.join(classpath, "bad")
                mask_dir = os.path.join(classpath, "ground_truth", "bad")
                
                if os.path.exists(bad_dir):
                    bad_files = sorted([f for f in os.listdir(bad_dir) if f.endswith(('.png', '.jpg'))])
                    imgpaths_per_class[classname]["bad"] = [os.path.join(bad_dir, f) for f in bad_files]
                    
                    for image_file in bad_files:
                        image_path = os.path.join(bad_dir, image_file)
                        
                        # Find corresponding mask
                        # Mask naming: image "005_regular.png" -> mask "005_regular_mask.png"
                        mask_file = image_file.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
                        mask_path = os.path.join(mask_dir, mask_file) if os.path.exists(mask_dir) else None
                        
                        if mask_path and not os.path.exists(mask_path):
                            mask_path = None
                        
                        data_tuple = (classname, "bad", image_path, mask_path)
                        data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate