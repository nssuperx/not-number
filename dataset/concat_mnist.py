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


# デバッグ用
if __name__ == "__main__":
    train, test = generate_mnist_with_dctbasis()
    print(f"MNIST + DCT Train dataset size: {len(train)}")
    print(f"MNIST + DCT Test dataset size: {len(test)}")
    img, label = train[-10]
    print(f"Image shape: {img.shape}, Label: {label}")

    mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())
    print(f"MNIST Train dataset size: {len(mnist_train)}")
    print(f"MNIST Test dataset size: {len(mnist_test)}")
    mnist_img, mnist_label = mnist_train[0]
    print(f"MNIST Image shape: {mnist_img.shape}, Label: {mnist_label}")

    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=64, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=64, num_workers=2)

    from dct_basis import visualize_dct_basis

    for imgs, labels in train_loader:
        print(f"Batch size: {imgs.shape[0]}, Image shape: {imgs.shape}, Labels shape: {labels.shape}")
        for img, label in zip(imgs, labels):
            print(f"Image shape: {img.shape}, Label: {label.item()}")
        visualize_dct_basis(imgs.reshape(8, 8, 28, 28), grid_rows=8)
        break
