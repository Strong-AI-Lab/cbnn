
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import pytorch_lightning as pl

# Default constants
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_SAVE_DIR = "./data/"


# Utils
class BaseDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(BaseDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split

    def _split_train_val_data(self, data):
        train, val = data_utils.random_split(data, [int(len(data) * (1 - self.train_val_split)), int(len(data) * self.train_val_split)])
        return train, val
    
    @classmethod
    def add_data_specific_args(cls, parent_parser):
        parser = parent_parser.add_argument_group("DataModule")
        parser.add_argument("--data_dir", type=str, default=DEFAULT_SAVE_DIR, help="Path to the data directory.")
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the data loader.")
        parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for the data loader.")
        parser.add_argument("--train_val_split", type=float, default=DEFAULT_VALIDATION_SPLIT, help="Validation split fraction.")
        return parent_parser

    def prepare_data(self):
        raise NotImplementedError()

    def setup(self, stage=None):
        raise NotImplementedError()

    def train_dataloader(self):
        return data_utils.DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return data_utils.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return data_utils.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)
    



# Data Modules
class MNISTDataModule(BaseDataModule):
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(MNISTDataModule, self).__init__(data_dir, batch_size, num_workers, train_val_split)
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
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(MNISTOODDataModule, self).__init__(data_dir, batch_size, num_workers, train_val_split)
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
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(CIFAR10DataModule, self).__init__(data_dir, batch_size, num_workers, train_val_split)
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
    def __init__(self, data_dir: str = DEFAULT_SAVE_DIR, batch_size: int = 32, num_workers: int = 4, train_val_split: int = DEFAULT_VALIDATION_SPLIT, **kwargs):
        super(CIFAR10OODDataModule, self).__init__(data_dir, batch_size, num_workers, train_val_split)
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




def acre(**kwargs):
    raise NotImplementedError("ACRE dataset not implemented yet.")

def acre_ood(**kwargs):
    raise NotImplementedError("ACRE dataset not implemented yet.")

def conceptarc(**kwargs):
    raise NotImplementedError("CONCEPTARC dataset not implemented yet.")

def conceptarc_ood(**kwargs):
    raise NotImplementedError("CONCEPTARC dataset not implemented yet.")

def raven(**kwargs):
    raise NotImplementedError("RAVEN dataset not implemented yet.")

def raven_ood(**kwargs):
    raise NotImplementedError("RAVEN dataset not implemented yet.")




DATASETS = {
    "MNIST": MNISTDataModule,
    "MNIST_OOD": MNISTOODDataModule,
    "CIFAR10": CIFAR10DataModule,
    "CIFAR10_OOD": CIFAR10OODDataModule,
    "ACRE": acre,
    "ACRE_OOD": acre_ood,
    "CONCEPTARC": conceptarc,
    "CONCEPTARC_OOD": conceptarc_ood,
    "RAVEN": raven,
    "RAVEN_OOD": raven_ood
}

def get_dataset(dataset_name: str, **kwargs):
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found.")
    return DATASETS[dataset_name](**kwargs)