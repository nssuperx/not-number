from typing import Optional, Callable
import torch
from torchvision import datasets, transforms

from dataset.dct_basis import DCTBasis


def generate_mnist_with_dctbasis(transform: Optional[Callable] = None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + DCT基底画像の重みづけの和のような画像のデータセットを返す

    :return: MNIST + DCT基底画像のデータセット
    :rtype: tuple[Dataset, Dataset]

    """

    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())
    dct_train = DCTBasis(len(mnist_train) // 10, transform=transform)
    dct_test = DCTBasis(len(mnist_test) // 10, transform=transform)
    train = torch.utils.data.ConcatDataset([mnist_train, dct_train])
    test = torch.utils.data.ConcatDataset([mnist_test, dct_test])

    return train, test


class NotNumberLabelFMNIST(torch.utils.data.Dataset):
    def __init__(self, data_size: int = 6000, train: bool = True, transform: Optional[Callable] = None) -> None:
        """FashionMNISTを数字ではないとするデータセット
        transformはデフォルトでtoTensor()的な動作が入る

        :param data_size: 使う画像数
        :type data_size: int
        :param train: datasets.FashionMNISTのtrain引数
        :type train: bool
        :param transform: datasets.FashionMNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
        :type transform: Optional[Callable]"""
        super().__init__()
        fmnist = datasets.FashionMNIST("../data", train=train, download=True)
        indices = torch.randperm(len(fmnist.data))[:data_size]
        self.data = fmnist.data[indices].unsqueeze(1)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        d = self.data[index].to(dtype=torch.float32) / 255.0  # toTensor的な動作
        if self.transform is not None:
            d = self.transform(d)
        return d, 10  # ラベルは適当で良い。0-9以外ならなんでもいい。数字ではないを表せたらいい。

    def __len__(self) -> int:
        return len(self.data)


def generate_mnist_with_fmnist(transform: Optional[Callable] = None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + FashionMNISTのデータセットを返す

    :param transform: datasets.MNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
    :type transform: Optional[Callable]
    :return: MNIST + FashionMNISTのデータセット(train, test)
    :rtype: tuple[Dataset, Dataset]
    """
    mnist_tf = transforms.ToTensor()
    if transform is not None:
        mnist_tf = transforms.Compose([mnist_tf, transform])
    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=mnist_tf)
    mnist_test = datasets.MNIST("../data", train=False, transform=mnist_tf)
    fmnist_train = NotNumberLabelFMNIST(data_size=len(mnist_train) // 10, train=True, transform=transform)
    fmnist_test = NotNumberLabelFMNIST(data_size=len(mnist_test) // 10, train=False, transform=transform)
    train = torch.utils.data.ConcatDataset([mnist_train, fmnist_train])
    test = torch.utils.data.ConcatDataset([mnist_test, fmnist_test])

    return train, test


class NotNumberLabelKMNIST(torch.utils.data.Dataset):
    def __init__(self, data_size: int = 6000, train: bool = True, transform: Optional[Callable] = None) -> None:
        """KMNISTを数字ではないとするデータセット
        transformはデフォルトでtoTensor()的な動作が入る

        :param data_size: 使う画像数
        :type data_size: int
        :param train: datasets.KMNISTのtrain引数
        :type train: bool
        :param transform: datasets.KMNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
        :type transform: Optional[Callable]"""
        super().__init__()
        kmnist = datasets.KMNIST("../data", train=train, download=True)
        indices = torch.randperm(len(kmnist.data))[:data_size]
        self.data = kmnist.data[indices].unsqueeze(1)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        d = self.data[index].to(dtype=torch.float32) / 255.0  # toTensor的な動作
        if self.transform is not None:
            d = self.transform(d)
        return d, 10  # ラベルは適当で良い。0-9以外ならなんでもいい。数字ではないを表せたらいい。

    def __len__(self) -> int:
        return len(self.data)


def generate_mnist_with_kmnist(transform: Optional[Callable] = None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + KMNISTのデータセットを返す

    :param transform: datasets.MNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
    :type transform: Optional[Callable]
    :return: MNIST + KMNISTのデータセット(train, test)
    :rtype: tuple[Dataset, Dataset]
    """
    mnist_tf = transforms.ToTensor()
    if transform is not None:
        mnist_tf = transforms.Compose([mnist_tf, transform])
    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=mnist_tf)
    mnist_test = datasets.MNIST("../data", train=False, transform=mnist_tf)
    kmnist_train = NotNumberLabelKMNIST(data_size=len(mnist_train) // 10, train=True, transform=transform)
    kmnist_test = NotNumberLabelKMNIST(data_size=len(mnist_test) // 10, train=False, transform=transform)
    train = torch.utils.data.ConcatDataset([mnist_train, kmnist_train])
    test = torch.utils.data.ConcatDataset([mnist_test, kmnist_test])

    return train, test


def generate_mnist_with_variousimg(transform: Optional[Callable] = None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + FashionMNIST + KMNISTのデータセットを返す

    :param transform: datasets.MNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
    :type transform: Optional[Callable]
    :return: MNIST + FashionMNIST + KMNISTのデータセット(train, test)
    :rtype: tuple[Dataset, Dataset]
    """
    mnist_tf = transforms.ToTensor()
    if transform is not None:
        mnist_tf = transforms.Compose([mnist_tf, transform])

    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=mnist_tf)
    mnist_test = datasets.MNIST("../data", train=False, transform=mnist_tf)
    fmnist_train = NotNumberLabelFMNIST(data_size=len(mnist_train) // 20, train=True, transform=transform)
    fmnist_test = NotNumberLabelFMNIST(data_size=len(mnist_test) // 20, train=False, transform=transform)
    kmnist_train = NotNumberLabelKMNIST(data_size=len(mnist_train) // 20, train=True, transform=transform)
    kmnist_test = NotNumberLabelKMNIST(data_size=len(mnist_test) // 20, train=False, transform=transform)

    train = torch.utils.data.ConcatDataset([mnist_train, fmnist_train, kmnist_train])
    test = torch.utils.data.ConcatDataset([mnist_test, fmnist_test, kmnist_test])

    return train, test
