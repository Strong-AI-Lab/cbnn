
from torchvision import datasets, transforms
import torch.utils.data as data_utils


# Default constants
VALIDATION_SPLIT = 0.2


# Utils
def _split_train_val_data(data, split=VALIDATION_SPLIT):
    train, val = data_utils.random_split(data, [int(len(data) * (1 - split)), int(len(data) * split)])
    return train, val




# Data loading methods
def mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("data", train=True, download=True, transform=transform)
    train, val = _split_train_val_data(train)
    test = datasets.MNIST("data", train=False, download=True, transform=transform)
    return train, val, test


def mnist_ood():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_shift = transforms.Compose([
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("data", train=True, download=True, transform=transform)
    train, val = _split_train_val_data(train)
    test = datasets.MNIST("data", train=False, download=True, transform=transform_shift)
    return train, val, test


def cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    train, val = _split_train_val_data(train)
    test = datasets.CIFAR10("data", train=False, download=True, transform=transform)
    return train, val, test


def cifar10_ood():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_shift = transforms.Compose([
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    train, val = _split_train_val_data(train)
    test = datasets.CIFAR10("data", train=False, download=True, transform=transform_shift)
    return train, val, test


def acre():
    raise NotImplementedError("ACRE dataset not implemented yet.")

def acre_ood():
    raise NotImplementedError("ACRE dataset not implemented yet.")

def conceptarc():
    raise NotImplementedError("CONCEPTARC dataset not implemented yet.")

def conceptarc_ood():
    raise NotImplementedError("CONCEPTARC dataset not implemented yet.")

def raven():
    raise NotImplementedError("RAVEN dataset not implemented yet.")

def raven_ood():
    raise NotImplementedError("RAVEN dataset not implemented yet.")




DATASETS = {
    "MNIST": mnist,
    "MNIST_OOD": mnist_ood,
    "CIFAR10": cifar10,
    "CIFAR10_OOD": cifar10_ood,
    "ACRE": acre,
    "ACRE_OOD": acre_ood,
    "CONCEPTARC": conceptarc,
    "CONCEPTARC_OOD": conceptarc_ood,
    "RAVEN": raven,
    "RAVEN_OOD": raven_ood
}