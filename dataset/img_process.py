import torch
from torch.nn.functional import sigmoid


# github copilotに書かせた


def minmax_normalize(imgs: torch.Tensor) -> torch.Tensor:
    """各画像ごとにmin-max正規化"""
    return (imgs - imgs.amin(dim=(-2, -1), keepdim=True)) / (
        imgs.amax(dim=(-2, -1), keepdim=True) - imgs.amin(dim=(-2, -1), keepdim=True) + 1e-8
    )


def binarize_imgs(imgs: torch.Tensor, n: int) -> torch.Tensor:
    """各画像ごとにn値化

    :param imgs: 画像テンソル
    :type imgs: Tensor
    :param n: n値化のレベル
    :type n: int
    :return: n値化された画像テンソル
    :rtype: Tensor
    """
    min_val = imgs.amin(dim=(-2, -1), keepdim=True)
    max_val = imgs.amax(dim=(-2, -1), keepdim=True)
    norm = (imgs - min_val) / (max_val - min_val + 1e-8)
    quantized = torch.floor(norm * n)
    quantized = quantized / (n - 1)
    return quantized


def preprocess_imgs(imgs: torch.Tensor) -> torch.Tensor:
    """前処理する。MNISTの画像に近づける
    MNISTは背景が0.0で文字が1.0
    色の分布のさせ方をはっきりさせる

    :param imgs: Description
    :type imgs:
    :return: Description
    :rtype: Tensor"""

    # 28x28の中心が1.0、外側が0.0以下になるグラデーション画像を生成
    # 0.0以下は0.0にclampする
    size = 28
    y = torch.arange(size).float()
    x = torch.arange(size).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    center = (size - 1) / 2
    dist = torch.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    # この辺は適当。縁あたりがMNIST同様に0.0くらいになればいい
    # 中心あたりが1.0になりやすい問題はある
    bg = 1.0 - (dist / (center * 1.2))
    processed_img = bg * imgs
    processed_img = sigmoid((processed_img - 0.3) * 15.0)

    return processed_img


if __name__ == "__main__":
    # デバッグ用
    imgs = torch.rand(10, 1, 28, 28)  # 10枚の28x28の画像テンソルを生成
    preprocessed_imgs = preprocess_imgs(imgs)
    print(preprocessed_imgs)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(preprocessed_imgs[0][0].numpy(), cmap="gray")
    plt.axis("off")
    plt.show()
