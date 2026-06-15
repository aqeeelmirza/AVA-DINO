# """
# Brain MRI Tumor Classification Dataset Loader
# """

# import os
# from enum import Enum
# import PIL
# import torch
# from torchvision import transforms
# import random

# _CLASSNAMES = ["01"]  # Brain MRI has only one class for anomaly detection

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]

# class DatasetSplit(Enum):
#     TRAIN = "Training"
#     VAL = "Testing"  # Using Testing as validation
#     TEST = "Testing"

# class Dataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         source,
#         classname,
#         resize=256,
#         imagesize=224,
#         split=DatasetSplit.TRAIN,
#         clip_transformer=None,
#         k_shot=0,
#         random_seed=42,
#         divide_num=1,
#         divide_iter=0,
#         **kwargs,
#     ):
#         super().__init__()
#         self.source = source
#         self.split = split
        
#         if classname == 'ALL':
#             self.classnames_to_use = _CLASSNAMES
#         else:
#             self.classnames_to_use = [classname]

#         self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        
#         if divide_num > 1:
#             self.data_to_iterate = self.sub_datasets(self.data_to_iterate, divide_num, divide_iter, random_seed)
        
#         if k_shot > 0:
#             torch.manual_seed(random_seed)
#             if k_shot >= len(self.data_to_iterate):
#                 pass
#             else:
#                 indices = torch.randint(0, len(self.data_to_iterate), (k_shot,))
#                 self.data_to_iterate = [self.data_to_iterate[i] for i in indices]
        
#         if clip_transformer is None:
#             self.transform_img = [
#                 transforms.Resize((resize, resize)),
#                 transforms.CenterCrop(imagesize),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#             ]
#             self.transform_img = transforms.Compose(self.transform_img)
#         else:
#             self.transform_img = clip_transformer

#         self.transform_mask = [
#             transforms.Resize((resize, resize)),
#             transforms.CenterCrop(imagesize),
#             transforms.ToTensor(),
#         ]
#         self.transform_mask = transforms.Compose(self.transform_mask)

#         self.imagesize = (3, imagesize, imagesize)

#     def sub_datasets(self, full_datasets, divide_num, divide_iter, random_seed=42):
#         if divide_num == 0:
#             return full_datasets
#         random.seed(random_seed)
        
#         id_dict = {}
#         for i in range(len(full_datasets)):
#             anomaly_type = full_datasets[i][1]  # "Normal" or "Anomaly"
#             if anomaly_type not in id_dict.keys():
#                 id_dict[anomaly_type] = []
#             id_dict[anomaly_type].append(i)
        
#         sub_id_list = []
#         for k in id_dict.keys():
#             type_id_list = id_dict[k]
#             random.shuffle(type_id_list)
#             devide_list = [type_id_list[i:i+divide_num] for i in range(0, len(type_id_list), divide_num)]
#             sub_list = [devide_list[i][divide_iter] for i in range(len(devide_list)) if len(devide_list[i])>divide_iter]
#             sub_id_list.extend(sub_list)
        
#         return [full_datasets[id] for id in sub_id_list]

#     def __getitem__(self, idx):
#         classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
#         image = PIL.Image.open(image_path).convert("RGB")
#         image = self.transform_img(image)

#         # No pixel-level masks for this dataset
#         mask = torch.zeros([1, *image.size()[1:]])

#         return {
#             "image": image,
#             "mask": mask,
#             "is_anomaly": int(anomaly != "Normal"),
#             "image_path": image_path,
#         }

#     def __len__(self):
#         return len(self.data_to_iterate)

#     def get_image_data(self):
#         data_to_iterate = []
        
#         for classname in self.classnames_to_use:
#             split_dir = os.path.join(self.source, self.split.value)
            
#             if not os.path.exists(split_dir):
#                 continue
            
#             # Define normal and anomaly categories
#             normal_category = 'notumor'
#             anomaly_categories = ['glioma', 'meningioma', 'pituitary']
            
#             # Load normal images
#             normal_dir = os.path.join(split_dir, normal_category)
#             if os.path.exists(normal_dir):
#                 normal_files = sorted([f for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.png'))])
#                 for img_file in normal_files:
#                     image_path = os.path.join(normal_dir, img_file)
#                     data_tuple = (classname, "Normal", image_path, None)
#                     data_to_iterate.append(data_tuple)
            
#             # Load anomaly images from all tumor types
#             for anomaly_cat in anomaly_categories:
#                 anomaly_dir = os.path.join(split_dir, anomaly_cat)
#                 if os.path.exists(anomaly_dir):
#                     anomaly_files = sorted([f for f in os.listdir(anomaly_dir) if f.endswith(('.jpg', '.png'))])
#                     for img_file in anomaly_files:
#                         image_path = os.path.join(anomaly_dir, img_file)
#                         data_tuple = (classname, "Anomaly", image_path, None)
#                         data_to_iterate.append(data_tuple)

#         return None, data_to_iterate

"""
Brain MRI Tumor Classification Dataset Loader
Updated to use MedSAM-generated masks for anomaly classes
"""

import os
from enum import Enum
import PIL
import torch
from torchvision import transforms
import random

_CLASSNAMES = ["01"]  # Brain MRI has only one class for anomaly detection

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DatasetSplit(Enum):
    TRAIN = "Training"
    VAL = "Testing"  # Using Testing as validation
    TEST = "Testing"

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

        # Load MedSAM-generated mask if available (for anomaly images)
        if mask_path is not None and os.path.exists(mask_path):
            mask = PIL.Image.open(mask_path).convert('L')
            mask = self.transform_mask(mask) > 0  # Binarize mask
        else:
            # For normal images or if mask doesn't exist, use zero mask
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": int(anomaly != "Normal"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []
        
        for classname in self.classnames_to_use:
            split_dir = os.path.join(self.source, self.split.value)
            
            if not os.path.exists(split_dir):
                continue
            
            # Define normal and anomaly categories
            normal_category = 'notumor'
            anomaly_categories = ['glioma', 'meningioma', 'pituitary']
            
            # Load normal images (no masks needed)
            normal_dir = os.path.join(split_dir, normal_category)
            if os.path.exists(normal_dir):
                normal_files = sorted([f for f in os.listdir(normal_dir) if f.endswith(('.jpg', '.png'))])
                for img_file in normal_files:
                    image_path = os.path.join(normal_dir, img_file)
                    # Normal images have no mask (None)
                    data_tuple = (classname, "Normal", image_path, None)
                    data_to_iterate.append(data_tuple)
            
            # Load anomaly images with MedSAM-generated masks
            for anomaly_cat in anomaly_categories:
                anomaly_dir = os.path.join(split_dir, anomaly_cat)
                mask_dir = os.path.join(self.source, "Masks", anomaly_cat)
                
                if os.path.exists(anomaly_dir):
                    anomaly_files = sorted([f for f in os.listdir(anomaly_dir) if f.endswith(('.jpg', '.png'))])
                    
                    for img_file in anomaly_files:
                        image_path = os.path.join(anomaly_dir, img_file)
                        
                        # Find corresponding MedSAM-generated mask
                        mask_file = img_file.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png')
                        mask_path = os.path.join(mask_dir, mask_file)
                        
                        # Check if mask exists, otherwise set to None
                        if not os.path.exists(mask_path):
                            mask_path = None
                        
                        # Anomaly images with mask path (or None if mask generation failed)
                        data_tuple = (classname, "Anomaly", image_path, mask_path)
                        data_to_iterate.append(data_tuple)

        return None, data_to_iterate