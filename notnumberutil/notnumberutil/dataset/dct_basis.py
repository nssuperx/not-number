from typing import Optional, Callable
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from .img_process import minmax_normalize, preprocess_imgs


# github copilotに書かせた


def dct_basis(size: int) -> torch.Tensor:
    """N x NのDCT基底行列を返す

    :param size: 基底画像の1辺のサイズ
    :type size: int
    :return: N x NのDCT基底行列
    :rtype: Tensor
    """
    x = np.arange(size)
    k = np.arange(size).reshape(-1, 1)
    alpha = np.sqrt(2.0 / size) * np.ones((size, 1))
    alpha[0, 0] = np.sqrt(1.0 / size)
    basis_1d = alpha * np.cos(np.pi * (2 * x + 1) * k / (2 * size))
    basis_2d = np.einsum("ik,jl->ijkl", basis_1d, basis_1d)
    return torch.from_numpy(basis_2d).float()


def visualize_dct_basis(imgs: torch.Tensor, grid_rows: int):
    # 基底をmake_gridで扱えるように変換
    imgs = imgs[:grid_rows, :grid_rows].reshape(-1, imgs.shape[-2], imgs.shape[-1]).unsqueeze(1)

    # make_gridでグリッド画像を作成
    grid_img = make_grid(imgs, nrow=grid_rows, normalize=False, padding=1)

    # matplotlibで表示
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()


def random_weighted_sum_from_basis(basis: torch.Tensor, N: int, img_num: int) -> torch.Tensor:
    """
    DCT基底画像テンソルからランダムにN個選び、ランダム重みで合成した画像をimg_num枚生成

    :param basis: DCT基底画像テンソル [B1, B2, H, W](例: [28, 28, 28, 28])
    :param N: 使う基底画像数
    :param img_num: 作る画像数
    :return: 合成画像テンソル [img_num, H, W]
    """
    B1, B2, H, W = basis.shape
    basis_flat = basis.reshape(-1, H, W)  # [B1*B2, H, W]
    total_basis = basis_flat.shape[0]
    # [img_num, N]のランダムなインデックス
    idx = torch.randint(0, total_basis, (img_num, N))
    # [img_num, N]のランダム重み
    weights = torch.randn(img_num, N)
    weights = weights / (weights.norm(dim=1, keepdim=True) + 1e-8)
    # [img_num, N, H, W]
    selected = basis_flat[idx]  # バッチインデックスで選択
    # [img_num, H, W] 合成
    imgs = (weights.unsqueeze(-1).unsqueeze(-1) * selected).sum(dim=1)
    return imgs  # [img_num, H, W]


class DCTBasis(torch.utils.data.Dataset):
    def __init__(self, data_num: int = 6000, transform: Optional[Callable] = None) -> None:
        super().__init__()
        base_size = 28  # mnistの画像サイズと同じ
        basis = dct_basis(base_size)
        basis = basis[:5, :5]  # いったん適当で。たぶん高周波成分は要らないと思う
        use_base = 10  # 増やしてもあまり意味ない。変化の少ないノイズ画像みたいになる
        data_imgs = random_weighted_sum_from_basis(basis, use_base, data_num)
        data_imgs = minmax_normalize(data_imgs)
        data_imgs = preprocess_imgs(data_imgs)  # これが必要かは分からない
        self.data = data_imgs.unsqueeze(1)  # [data_num, 1, H, W]にする(torchvisionのmnistと同じ形にするため)
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        d = self.data[index]
        if self.transform is not None:
            d = self.transform(d)
        return d, 10  # ラベルは適当で良い。0-9以外ならなんでもいい。数字ではないを表せたらいい。

    def __len__(self) -> int:
        return len(self.data)


# デバッグ用
if __name__ == "__main__":
    N = 28
    basis = dct_basis(N)

    basis = basis[:5, :5]
    random_imgs = random_weighted_sum_from_basis(basis, 10, 100)
    # min-max正規化
    imgs = minmax_normalize(random_imgs)

    # 画像をn値化
    # imgs = binarize_imgs(basis, 10)

    # make_gridでグリッド画像を作成
    imgs = imgs.unsqueeze(1)  # [100, 1, 28, 28]
    grid_img = make_grid(imgs, nrow=10, normalize=False, padding=1)

    # matplotlibで表示
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.axis("off")
    plt.show()
