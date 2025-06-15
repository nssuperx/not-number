from typing import Optional, Callable
import torch
from torchvision import datasets, transforms

from dataset.dct_basis import DCTBasis
from dataset.img_process import preprocess_imgs


def generate_mnist_with_dctbasis(
    root: str, transform: Optional[Callable] = None, train_not_number_size: int = 6000, test_not_number_size: int = 1000
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + DCT基底画像の重みづけの和のような画像のデータセットを返す

    :param root: データセットの保存先
    :type root: str
    :param transform: datasets.MNISTで設定するtransformのようなもの。toTensor()が入る
    :type transform: Optional[Callable]
    :return: MNIST + DCT基底画像のデータセット
    :rtype: tuple[Dataset, Dataset]"""
    mnist_train = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root, train=False, transform=transforms.ToTensor())
    dct_train = DCTBasis(train_not_number_size, transform=transform)
    dct_test = DCTBasis(test_not_number_size, transform=transform)
    train = torch.utils.data.ConcatDataset([mnist_train, dct_train])
    test = torch.utils.data.ConcatDataset([mnist_test, dct_test])

    return train, test


class NotNumberLabelFMNIST(torch.utils.data.Dataset):
    def __init__(self, root: str, data_size: int = 6000, train: bool = True, transform: Optional[Callable] = None) -> None:
        """FashionMNISTを数字ではないとするデータセット
        transformはデフォルトでtoTensor()的な動作が入る

        :param root: データセットの保存先
        :type root: str
        :param data_size: 使う画像数
        :type data_size: int
        :param train: datasets.FashionMNISTのtrain引数
        :type train: bool
        :param transform: datasets.FashionMNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
        :type transform: Optional[Callable]"""
        super().__init__()
        fmnist = datasets.FashionMNIST(root, train=train, download=True)
        indices = torch.randperm(len(fmnist.data))[:data_size]
        self.data = fmnist.data[indices].unsqueeze(1).to(dtype=torch.float32) / 255.0  # toTensor的な動作
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        d = self.data[index]
        if self.transform is not None:
            d = self.transform(d)
        return d, 10  # ラベルは適当で良い。0-9以外ならなんでもいい。数字ではないを表せたらいい。

    def __len__(self) -> int:
        return len(self.data)


class NotNumberLabelKMNIST(torch.utils.data.Dataset):
    def __init__(self, root: str, data_size: int = 6000, train: bool = True, transform: Optional[Callable] = None) -> None:
        """KMNISTを数字ではないとするデータセット
        transformはデフォルトでtoTensor()的な動作が入る

        :param root: データセットの保存先
        :type root: str
        :param data_size: 使う画像数
        :type data_size: int
        :param train: datasets.KMNISTのtrain引数
        :type train: bool
        :param transform: datasets.KMNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
        :type transform: Optional[Callable]"""
        super().__init__()
        kmnist = datasets.KMNIST(root, train=train, download=True)
        indices = torch.randperm(len(kmnist.data))[:data_size]
        self.data = kmnist.data[indices].unsqueeze(1).to(dtype=torch.float32) / 255.0  # toTensor的な動作
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        d = self.data[index]
        if self.transform is not None:
            d = self.transform(d)
        return d, 10  # ラベルは適当で良い。0-9以外ならなんでもいい。数字ではないを表せたらいい。

    def __len__(self) -> int:
        return len(self.data)


class NotNumberLabelNoise(torch.utils.data.Dataset):
    def __init__(self, data_size: int = 6000, transform: Optional[Callable] = None) -> None:
        """ノイズ画像を数字ではないとするデータセット
        transformはデフォルトでtoTensor()的な動作が入る

        :param data_size: 使う画像数
        :type data_size: int
        :param transform: datasets.FashionMNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
        :type transform: Optional[Callable]"""
        super().__init__()
        d = torch.rand((data_size, 1, 28, 28), dtype=torch.float32)  # [0,1)
        self.data = preprocess_imgs(d)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        d = self.data[index]
        if self.transform is not None:
            d = self.transform(d)
        return d, 10  # ラベルは適当で良い。0-9以外ならなんでもいい。数字ではないを表せたらいい。

    def __len__(self) -> int:
        return len(self.data)


def generate_mnist_with_fmnist(
    root: str, transform: Optional[Callable] = None, train_not_number_size: int = 6000, test_not_number_size: int = 1000
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + FashionMNISTのデータセットを返す

    :param root: データセットの保存先
    :type root: str
    :param transform: datasets.MNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
    :type transform: Optional[Callable]
    :return: MNIST + FashionMNISTのデータセット(train, test)
    :rtype: tuple[Dataset, Dataset]"""
    mnist_tf = transforms.ToTensor()
    if transform is not None:
        mnist_tf = transforms.Compose([mnist_tf, transform])
    mnist_train = datasets.MNIST(root, train=True, download=True, transform=mnist_tf)
    mnist_test = datasets.MNIST(root, train=False, transform=mnist_tf)
    fmnist_train = NotNumberLabelFMNIST(root, data_size=train_not_number_size, train=True, transform=transform)
    fmnist_test = NotNumberLabelFMNIST(root, data_size=test_not_number_size, train=False, transform=transform)
    train = torch.utils.data.ConcatDataset([mnist_train, fmnist_train])
    test = torch.utils.data.ConcatDataset([mnist_test, fmnist_test])

    return train, test


def generate_mnist_with_kmnist(
    root: str, transform: Optional[Callable] = None, train_not_number_size: int = 6000, test_not_number_size: int = 1000
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + KMNISTのデータセットを返す

    :param root: データセットの保存先
    :type root: str
    :param transform: datasets.MNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
    :type transform: Optional[Callable]
    :return: MNIST + KMNISTのデータセット(train, test)
    :rtype: tuple[Dataset, Dataset]"""
    mnist_tf = transforms.ToTensor()
    if transform is not None:
        mnist_tf = transforms.Compose([mnist_tf, transform])
    mnist_train = datasets.MNIST(root, train=True, download=True, transform=mnist_tf)
    mnist_test = datasets.MNIST(root, train=False, transform=mnist_tf)
    kmnist_train = NotNumberLabelKMNIST(root, data_size=train_not_number_size, train=True, transform=transform)
    kmnist_test = NotNumberLabelKMNIST(root, data_size=test_not_number_size, train=False, transform=transform)
    train = torch.utils.data.ConcatDataset([mnist_train, kmnist_train])
    test = torch.utils.data.ConcatDataset([mnist_test, kmnist_test])

    return train, test


def generate_mnist_with_variousimg(
    root: str, transform: Optional[Callable] = None, train_not_number_size: int = 6000, test_not_number_size: int = 1000
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + FashionMNIST + KMNISTのデータセットを返す

    :param transform: datasets.MNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
    :type transform: Optional[Callable]
    :return: MNIST + FashionMNIST + KMNISTのデータセット(train, test)
    :rtype: tuple[Dataset, Dataset]"""
    mnist_tf = transforms.ToTensor()
    if transform is not None:
        mnist_tf = transforms.Compose([mnist_tf, transform])

    mnist_train = datasets.MNIST(root, train=True, download=True, transform=mnist_tf)
    mnist_test = datasets.MNIST(root, train=False, transform=mnist_tf)
    fmnist_train = NotNumberLabelFMNIST(root, data_size=train_not_number_size // 2, train=True, transform=transform)
    fmnist_test = NotNumberLabelFMNIST(root, data_size=test_not_number_size // 2, train=False, transform=transform)
    kmnist_train = NotNumberLabelKMNIST(root, data_size=train_not_number_size // 2, train=True, transform=transform)
    kmnist_test = NotNumberLabelKMNIST(root, data_size=test_not_number_size // 2, train=False, transform=transform)
    train = torch.utils.data.ConcatDataset([mnist_train, fmnist_train, kmnist_train])
    test = torch.utils.data.ConcatDataset([mnist_test, fmnist_test, kmnist_test])

    return train, test


def generate_mnist_with_noise(
    root: str, transform: Optional[Callable] = None, train_not_number_size: int = 6000, test_not_number_size: int = 1000
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + ノイズ画像のデータセットを返す

    :param root: データセットの保存先
    :type root: str
    :param transform: datasets.MNISTで設定するtransformのようなもの。デフォルトでtoTensor()的な動作が入る
    :type transform: Optional[Callable]
    :return: MNIST + ノイズ画像のデータセット(train, test)
    :rtype: tuple[Dataset, Dataset]"""
    mnist_tf = transforms.ToTensor()
    if transform is not None:
        mnist_tf = transforms.Compose([mnist_tf, transform])
    mnist_train = datasets.MNIST(root, train=True, download=True, transform=mnist_tf)
    mnist_test = datasets.MNIST(root, train=False, transform=mnist_tf)
    noise_train = NotNumberLabelNoise(data_size=train_not_number_size, transform=transform)
    noise_test = NotNumberLabelNoise(data_size=test_not_number_size, transform=transform)
    train = torch.utils.data.ConcatDataset([mnist_train, noise_train])
    test = torch.utils.data.ConcatDataset([mnist_test, noise_test])

    return train, test
