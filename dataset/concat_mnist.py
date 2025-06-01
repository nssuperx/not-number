import torch
from torchvision import datasets, transforms

from dct_basis import DCTBasis


def generate_mnist_with_dctbasis() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + DCT基底画像の重みづけの和のような画像のデータセットを返す

    :return: MNIST + DCT基底画像のデータセット
    :rtype: tuple[Dataset, Dataset]

    """

    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())
    dct_train = DCTBasis(len(mnist_train) // 10)
    dct_test = DCTBasis(len(mnist_test) // 10)
    train = torch.utils.data.ConcatDataset([mnist_train, dct_train])
    test = torch.utils.data.ConcatDataset([mnist_test, dct_test])

    return train, test


class NotNumberLabelFMNIST(torch.utils.data.Dataset):
    def __init__(self, data_size: int = 7000, train: bool = True) -> None:
        super().__init__()
        fmnist = datasets.FashionMNIST("../data", train=train, download=True, transform=transforms.ToTensor())
        indices = torch.randperm(len(fmnist.data))[:data_size]
        self.data = fmnist.data[indices].unsqueeze(1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.data[index], 10  # ラベルは適当で良い。0-9以外ならなんでもいい。数字ではないを表せたらいい。

    def __len__(self) -> int:
        return len(self.data)


def generate_mnist_with_fmnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """MNIST + FashionMNISTのデータセットを返す

    :return: MNIST + FashionMNISTのデータセット(train, test)
    :rtype: tuple[Dataset, Dataset]

    """
    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())
    fmnist_train = NotNumberLabelFMNIST(data_size=len(mnist_train) // 10, train=True)
    fmnist_test = NotNumberLabelFMNIST(data_size=len(mnist_test) // 10, train=False)
    train = torch.utils.data.ConcatDataset([mnist_train, fmnist_train])
    test = torch.utils.data.ConcatDataset([mnist_test, fmnist_test])

    return train, test
