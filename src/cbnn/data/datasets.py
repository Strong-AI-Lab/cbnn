
import os
import json
from typing import Optional
import numpy as np
import requests
import tqdm

import torch
from torchvision import datasets, transforms, io
import torchvision.transforms.v2 as transforms_v2
import torch.utils.data as data_utils
import pytorch_lightning as pl

# Default constants
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_SAVE_DIR = "./data/"


# Utils
class BaseDataset(data_utils.Dataset):
    def __init__(self, x, y, x_transform=None, y_transform=None):
        self.x = x
        self.y = y
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.x_transform:
            x = self.x_transform(x)

        if self.y_transform:
            y = self.y_transform(y)

        return x, y

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, split: Optional[str] = None, mode: str = "inference", batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(BaseDataModule, self).__init__()        
        self.data_dir = data_dir
        self.distribution_split = split
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.test_data = []
        self.train_data = []
        self.val_data = []

    def _split_train_val_data(self, data):
        train, val = data_utils.random_split(data, [int(len(data) * (1 - self.train_val_split)), int(len(data) * self.train_val_split)])
        return train, val
    
    @classmethod
    def add_data_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--data_dir", type=str, default=DEFAULT_SAVE_DIR, help="Path to the data directory.")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the data loader.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the data loader.")
        parser.add_argument("--train_val_split", type=float, default=DEFAULT_VALIDATION_SPLIT, help="Optional. Validation split fraction. Is used for datasets with no validation set provided.")
        parser.add_argument("--split", type=str, default=None, help="Optional. Split of the dataset to use if the dataset contains multiple splits (e.g. i.i.d and o.o.d).")
        parser.add_argument("--mode", type=str, default="inference", help="Optional. Mode of the data module (inference/generation). Used only if the dataset contains multiple images per input. In inference mode, an extra dimension is added to the input to represent the sequence of images. In generation mode, the input is a single image and the output is an auxiliary task requiring a single image.")
        return parent_parser

    def prepare_data(self):
        raise NotImplementedError()

    def setup(self, stage=None):
        raise NotImplementedError()

    def train_dataloader(self):
        return data_utils.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return data_utils.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return data_utils.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    



# Data Modules
class MNISTDataModule(BaseDataModule):
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, split: Optional[str] = None, mode: str = "inference", batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(MNISTDataModule, self).__init__(data_dir, split, mode, batch_size, num_workers, train_val_split, **kwargs)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        # Download the dataset
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            train = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = self._split_train_val_data(train)
        elif stage == "test":
            self.test_data = datasets.MNIST(self.data_dir, train=False, transform=self.transform)



class MNISTOODDataModule(MNISTDataModule):
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, split: Optional[str] = None, mode: str = "inference", batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(MNISTOODDataModule, self).__init__(data_dir, split, mode, batch_size, num_workers, train_val_split, **kwargs)
        self.transform_shift = transforms.Compose([
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        if stage == "fit":
            train = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = self._split_train_val_data(train)
        elif stage == "test":
            self.test_data = datasets.MNIST(self.data_dir, train=False, transform=self.transform_shift)



class CIFAR10DataModule(BaseDataModule):
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, split: Optional[str] = None, mode: str = "inference", batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(CIFAR10DataModule, self).__init__(data_dir, split, mode, batch_size, num_workers, train_val_split, **kwargs)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def prepare_data(self):
        # Download the dataset
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit":
            train = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = self._split_train_val_data(train)
        elif stage == "test":
            self.test_data = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)



class CIFAR10OODDataModule(CIFAR10DataModule):
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, split: Optional[str] = None, mode: str = "inference", batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(CIFAR10OODDataModule, self).__init__(data_dir, split, mode, batch_size, num_workers, train_val_split, **kwargs)
        self.transform_shift = transforms.Compose([
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        if stage == "fit":
            train = datasets.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.train_data, self.val_data = self._split_train_val_data(train)
        elif stage == "test":
            self.test_data = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform_shift)



class ACREDataModule(BaseDataModule):

    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, split: Optional[str] = None, mode: str = "inference", batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        if split is not None:
            data_dir = os.path.join(data_dir, split)

        super(ACREDataModule, self).__init__(data_dir, split, mode, batch_size, num_workers, train_val_split, **kwargs)
        self.load_image_tensor = transforms.Compose([
            io.read_image, # directly return Tensors in range [0,255] so we normalize from this range
            transforms_v2.Lambda(lambda x: x[..., :3,:,:]), # Remove alpha channel
            transforms_v2.ToDtype(torch.float32),
            # transforms_v2.Normalize((131.6699, 126.6359, 125.6257, 255.0000), (36.4253, 29.8901, 30.2831,  0.0001)),
            transforms_v2.Normalize((131.6699, 126.6359, 125.6257), (36.4253, 29.8901, 30.2831)), # Mean and std of the dataset based on the IID training set
            transforms_v2.Pad([0,40,0,40]), # [320x240] --> [320x320]
            transforms_v2.Resize((256, 256)) # [320x320] --> [256x256]
        ])

    def _build_image_tensor_sequence(self, image_files):
        image_sequence = []
        for image_file in image_files:
            image = self.load_image_tensor(image_file)
            image_sequence.append(image)
        return torch.stack(image_sequence)


    def prepare_data(self):
        assert os.path.exists(self.data_dir), f"Dataset not found at {self.data_dir}."
        assert os.path.exists(os.path.join(self.data_dir, 'config/')), f"Config not found at {os.path.join(self.data_dir, 'config/')}."
        assert os.path.exists(os.path.join(self.data_dir, 'images/')), f"Images not found at {os.path.join(self.data_dir, 'images/')}."
        assert os.path.exists(os.path.join(self.data_dir, 'config/', 'test.json')), f"Test config not found at {os.path.join(self.data_dir, 'config/', 'test.json')}."
        assert os.path.exists(os.path.join(self.data_dir, 'config/', 'train.json')), f"Train config not found at {os.path.join(self.data_dir, 'config/', 'train.json')}."
        assert os.path.exists(os.path.join(self.data_dir, 'config/', 'val.json')), f"Val config not found at {os.path.join(self.data_dir, 'config/', 'val.json')}."

    def _setup_split(self, split):
        config_file = os.path.join(self.data_dir, 'config/', f'{split}.json')
        images_dir = os.path.join(self.data_dir, 'images/')
        images_files = os.listdir(images_dir)

        with open(config_file, 'r') as f:
            samples = json.load(f)

        img_idx = 0
        x = []
        y = []
        for sample in samples:
            offset = len(sample)
            context = [c for c in sample if c['light_state'] != 'no']
            target = [c for c in sample if c['light_state'] == 'no']

            nb_context = len(context)
            nb_target = len(target)


            if self.mode == "inference":
                context_imgs = [os.path.join(images_dir, img) for img in images_files[img_idx:img_idx+nb_context]]
                for i in range(nb_target):
                    x.append(context_imgs + [os.path.join(images_dir, images_files[img_idx + nb_context + i])])
                    y.append(int(target[i]['label']))
            else:
                for i in range(len(context)):
                    x.append(os.path.join(images_dir, images_files[img_idx+i]))
                    y.append(int(context[i]['light_state'] == 'on'))
                for j in range(len(target)):
                    x.append(os.path.join(images_dir, images_files[img_idx + nb_context + j]))
                    y.append(int(target[j]['label']))

            img_idx += offset

        if self.mode == "inference":
            return BaseDataset(x, y, self._build_image_tensor_sequence)
        else:
            return BaseDataset(x, y, self.load_image_tensor)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_data = self._setup_split("train")
            self.val_data = self._setup_split("val")
        elif stage == "test":
            self.test_data = self._setup_split("test")
        


class RAVENDataModule(BaseDataModule):
    DISTRIB = [
        "center_single",
        "distribute_four",
        "distribute_nine",
        "in_center_single_out_center_single",
        "in_distribute_four_out_center_single",
        "left_center_single_right_center_single",
        "up_center_single_down_center_single"
    ]

    META_TARGET_FORMAT = [
        "Constant", 
        "Progression", 
        "Arithmetic", 
        "Distribute_Three", 
        "Number", 
        "Position", 
        "Type", 
        "Size", 
        "Color"
    ] # from https://github.com/WellyZhang/RAVEN/blob/master/src/dataset/const.py

    DISTRIB_SPLITS = {
        "IID" : {
            "train" : ["center_single", "distribute_four", "distribute_nine"],
            "val" : ["center_single", "distribute_four", "distribute_nine"],
            "test" : ["center_single", "distribute_four", "distribute_nine"]
        },
        "IID_SMALL" : {
            "train" : ["center_single"],
            "val" : ["center_single"],
            "test" : ["center_single"]
        },
        "OOD" : {
            "train" : ["center_single", "distribute_four", "distribute_nine"],
            "val" : ["center_single", "distribute_four", "distribute_nine"],
            "test" : ["in_center_single_out_center_single", "in_distribute_four_out_center_single", "left_center_single_right_center_single", "up_center_single_down_center_single"]
        },
        "OOD_SMALL" : {
            "train" : ["center_single"],
            "val" : ["center_single"],
            "test" : ["distribute_four", "distribute_nine"]
        },
        "OOD_TRANSFER" : {
            "train" : ["center_single", "distribute_four", "in_center_single_out_center_single"],
            "val" : ["center_single", "distribute_four", "in_center_single_out_center_single"],
            "test" : ["in_distribute_four_out_center_single"]
        } 
    }


    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, split: Optional[str] = None, mode: str = "inference", batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(RAVENDataModule, self).__init__(data_dir, split, mode, batch_size, num_workers, train_val_split, **kwargs)
        self.image_transform_fn = transforms_v2.Compose([
            transforms.ToTensor(), # [0, 255] -> [0, 1]
            transforms_v2.ToDtype(torch.float32),
            transforms_v2.Normalize((0.8522,), (0.2990,)),
            transforms_v2.Resize((128, 128)) # [160x160] --> [128x128]
        ])

        if mode == "inference":
            self.transform = self._load_image_tensor_sequence
            self.target_transform = self._load_target
        else:
            self.transform = self._load_image_tensor_single
            self.target_transform = lambda x : 0

    def _load_image_tensor_sequence(self, image_file):
        data = np.load(image_file)
        images = [self.image_transform_fn(i) for i in data['image']]
        return torch.stack(images)
    
    def _load_target(self, target_file):
        data = np.load(target_file)
        return int(data['target'])
    
    def _load_image_tensor_single(self, image_file):
        data = np.load(image_file)
        return self.image_transform_fn(data['image'][0])


    def prepare_data(self):
        assert os.path.exists(self.data_dir), f"Dataset not found at {self.data_dir}."
        for distrib in self.DISTRIB:
            assert os.path.exists(os.path.join(self.data_dir, distrib)), f"Directory {distrib} not found at {self.data_dir}."

    def _setup_split(self, split):
        distrib_split = self.DISTRIB_SPLITS[self.distribution_split][split]
        x = []
        y = []
        for distrib in distrib_split:
            distrib_dir = os.path.join(self.data_dir, distrib)

            # List all files in the directory, filter .npz files and train, val and test files
            distrib_files = [f for f in os.listdir(distrib_dir) if f.endswith(f'_{split}.npz')]

            # Load data
            for distrib_file in distrib_files:
                distrib_file_path = os.path.join(distrib_dir, distrib_file)
                x.append(distrib_file_path)
                y.append(distrib_file_path)
                
        return BaseDataset(x, y, self.transform, self.target_transform)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_data = self._setup_split("train")
            self.val_data = self._setup_split("val")
        elif stage == "test":
            self.test_data = self._setup_split("test")



class ConceptARCDataModule(BaseDataModule):
    CONCEPTS = [
        "AboveBelow",
        "Center",
        "CleanUp",
        "CompleteShape",
        "Copy",
        "Count",
        "ExtendToBoundary",
        "ExtractObjects",
        "FilledNotFilled",
        "HorizontalVertical",
        "InsideOutside",
        "MoveToBoundary",
        "Order",
        "SameDifferent",
        "TopBottom2D",
        "TopBottom3D"
    ]
    IDS = {
        "train": [1, 2, 3, 4, 5, 6],
        "val": [7, 8],
        "test": [9, 10]
    }

    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, split: Optional[str] = None, mode: str = "inference", batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(ConceptARCDataModule, self).__init__(data_dir, split, mode, batch_size, num_workers, train_val_split, **kwargs)

        self.img_transform = transforms_v2.Compose([
            transforms_v2.ToDtype(torch.float32),
            transforms_v2.CenterCrop(32),
            transforms_v2.Resize((128, 128), interpolation=transforms_v2.InterpolationMode.BICUBIC),
        ])
        self.random_img_transform = transforms_v2.Compose([
            transforms_v2.ToDtype(torch.float32),
            transforms_v2.CenterCrop(32),
            transforms_v2.Resize((128, 128), interpolation=transforms_v2.InterpolationMode.BICUBIC),
            transforms_v2.RandomHorizontalFlip(),
            transforms_v2.RandomVerticalFlip(),
            transforms_v2.RandomRotation(30),
            transforms_v2.RandomAffine(0, translate=(0.1, 0.1))
        ])
        if mode == "inference":
            self.transform = self.img_transform
        else:
            self.transform = self.random_img_transform

    def _single_transform(self, x, num_classes=10): # Single image transform. [H x W] --> [C x 32 x 32]
        x = torch.tensor(x)
        x = torch.nn.functional.one_hot(x.long(), num_classes=num_classes) # [H x W] --> [H x W x C]
        x = x.permute(2, 0, 1) # [H x W x C] --> [C x H x W]
        return self.transform(x) # [C x H x W] --> [C x 32 x 32]

    def _multiple_context_transform(self, x):
        return torch.stack([self._single_transform(i) for i in x])


    def _download_data(self, concept : str, i : int, data_dir : Optional[str] =  None):
        url = f"https://raw.githubusercontent.com/victorvikram/ConceptARC/main/corpus/{concept}/{concept}{int(i)}.json"
        
        r = requests.get(url)
        data = json.loads(r.text)

        # Save data
        if data_dir is None:
            data_dir = self.data_dir
        
        data_dir = os.path.join(data_dir, concept)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        with open(os.path.join(data_dir, f"{concept}{int(i)}.json"), 'w') as f:
            json.dump(data, f)

    def _load_data(self, concept : str, i : int):
        # Load file
        concept_dir = os.path.join(self.data_dir, concept)
        concept_file = os.path.join(concept_dir, f"{concept}{int(i)}.json")
        with open(concept_file, 'r') as f:
            json_data = json.load(f)
        
        # Build data samples
        x = []
        y = []
        for key in json_data.keys(): # ['train', 'test']
            for sample in json_data[key]:
                if self.mode == "inference": # TODO: fix problem in inference mode. Number of images in the context can vary!
                    if key == 'train':
                        if len(x) == 0: # Initialisation of concept sample
                            x.append([])
                        
                        x[-1].append(sample['input']) # Context sample input-output pairs (input and output are provided to the context)
                        x[-1].append(sample['output'])
                    else:
                        x[-1].append(sample['input']) # Test sample input-output pairs (only input is provided to the context, output is the target)
                        y.append(sample['output'])

                else:
                    x.append(sample['input'])
                    x.append(sample['output'])
                    y.append(0)
                    y.append(0)

        return x, y


    def prepare_data(self):
        for concept in tqdm.tqdm(self.CONCEPTS, position=0):
            for i in tqdm.tqdm(range(1, 11), position=1, leave=False):
                if not os.path.exists(os.path.join(self.data_dir, concept, f"{concept}{int(i)}.json")):
                    self._download_data(concept, i)
    
    def _setup_split(self, split):
        x = []
        y = []
        for concept in self.CONCEPTS:
            for i in self.IDS[split]:
                concept_x, concept_y = self._load_data(concept, i)
                x += concept_x
                y += concept_y

        if self.mode == "inference":
            return BaseDataset(x, y, self._multiple_context_transform, self._single_transform)
        else:
            return BaseDataset(x, y, self._single_transform) # No target transform during generation mode as y is not used

    def setup(self, stage=None):
        if stage == "fit":
            self.train_data = self._setup_split("train")
            self.val_data = self._setup_split("val")
        elif stage == "test":
            self.test_data = self._setup_split("test")





DATASETS = {
    "MNIST": MNISTDataModule,
    "MNIST_OOD": MNISTOODDataModule,
    "CIFAR10": CIFAR10DataModule,
    "CIFAR10_OOD": CIFAR10OODDataModule,
    "ACRE": ACREDataModule,
    "CONCEPTARC": ConceptARCDataModule,
    "RAVEN": RAVENDataModule
}

def get_dataset(dataset_name: str, **kwargs):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found.")
    return DATASETS[dataset_name](**kwargs)