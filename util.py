import torch
import matplotlib.pyplot as plt


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        try:
            import torch_directml  # pyright: ignore[reportMissingImports]

            device = torch.device(torch_directml.device())
        except ImportError:
            pass

    return device


def show_accuracy_glaph(test_classes: dict[int, int], history: list[list[int]], path: str = "") -> None:
    epochs = range(1, len(history) + 1)
    test_size = sum(test_classes.values())
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [sum(c) / test_size for c in history])
    plt.title("Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    history = [list(x) for x in zip(*history)]
    for i in range(len(test_classes)):  # 0から始めたい
        linestyle = "-" if i == 10 else "--"
        plt.plot(epochs, [v / test_classes[i] for v in history[i]], label=f"{i}", linestyle=linestyle)
    plt.title("Accuracy per Class")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()
