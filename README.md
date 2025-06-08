# not-number

「MNIST + 手書き数字文字ではない適当な画像」を学習させて、数字ではない画像を分類できるか実験

## 環境構築

- [uv](https://docs.astral.sh/uv/)

CPU

```shell
uv sync --extra cpu
```

DirectML

```shell
uv sync --extra directml
```

Windows ROCm: <https://github.com/scottt/rocm-TheRock/releases>

```shell
uv sync --extra win-rocm
```
