from .interface import MNIST, FMNIST, SVHN, CIFAR10, CIFAR100
from .mnist import MNIST_Dataset

def load_dataset(dataset_name, normal_class, ratio):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'fashion-mnist', 'svhn', 'cifar10', 'cifar100')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST(normal_class=normal_class, ratio=ratio)
    if dataset_name == 'fashion-mnist':
        dataset = FMNIST(normal_class=normal_class, ratio=ratio)
    if dataset_name == 'svhn':
        dataset = SVHN(normal_class=normal_class, ratio=ratio)
    if dataset_name == 'cifar10':
        dataset = CIFAR10(normal_class=normal_class, ratio=ratio)
    if dataset_name == 'cifar100':
        dataset = CIFAR100(normal_class=normal_class, ratio=ratio)

    return dataset
