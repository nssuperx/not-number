import os
import sys
from collections import Counter
from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.concat_mnist import generate_mnist_with_variousimg
from util import get_device, show_accuracy_glaph, print_exp_param_mdtable

# https://github.com/pytorch/examples/blob/main/mnist/main.py


class CNN(nn.Module):
    def __init__(self, classes: int):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ExpParams(NamedTuple):
    not_number_train_size: int
    not_number_test_size: int
    train_size: int
    test_size: int
    batch_size: int
    epochs: int
    learning_rate: float


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    return loss.item()


def test(model: nn.Module, test_loader: torch.utils.data.DataLoader, num_classes: int, device: torch.device) -> tuple[float, list[int]]:
    model.eval()
    test_loss = 0
    correct = [0 for _ in range(num_classes)]
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output: torch.Tensor = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1)  # shape: (batch_size,)
            # クラスごとに正解数をカウント
            for cls in range(num_classes):
                correct[cls] += ((pred == cls) & (target == cls)).sum().item()
    test_loss /= len(test_loader.dataset)  # pyright: ignore[reportArgumentType]
    return test_loss, correct


def run(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    result_fig_path: str,
) -> None:
    device = get_device()
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=4)

    test_classes = Counter(label for _, label in test_dataset)
    test_size = test_classes.total()

    model = CNN(len(test_classes.keys())).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    correct_classes_history: list[list[int]] = []
    for epoch in range(epochs):
        train(model, train_loader, optimizer, device)
        loss, correct = test(model, test_loader, len(test_classes.keys()), device)
        sc = sum(correct)
        correct_classes_history.append(correct)
        print(f"Epoch {epoch + 1:2d}, Test Loss: {loss:.4f}, Correct: {sc}/{test_size}, Accuracy: {sc / test_size * 100:.2f}%")

    last_correct = correct_classes_history[-1]
    class_acc = (f"{c}: {(last_correct[i] / test_classes[c] * 100):.2f}%" for i, c in enumerate(sorted(test_classes.keys())))
    print("Accuracy:", ", ".join(class_acc))
    show_accuracy_glaph(test_classes, correct_classes_history, path=result_fig_path)


def setup_params(not_number_train_size: int, not_number_test_size: int, batch_size: int, epochs: int, learning_rate: float) -> ExpParams:
    return ExpParams(
        not_number_train_size=not_number_train_size,
        not_number_test_size=not_number_test_size,
        train_size=60000 + not_number_train_size,
        test_size=10000 + not_number_test_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    device = get_device()

    print("mnist")
    params = setup_params(0, 0, 100, 10, 0.05)
    print_exp_param_mdtable(params._asdict())
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=tf)
    test_dataset = datasets.MNIST("../data", train=False, transform=tf)
    run(train_dataset, test_dataset, params.batch_size, params.epochs, params.learning_rate, result_fig_path="imgs/mnist.png")

    print("not-number")
    params = setup_params(60000, 10000, 100, 10, 0.05)
    print_exp_param_mdtable(params._asdict())
    train_dataset, test_dataset = generate_mnist_with_variousimg(
        "../data", transforms.Normalize((0.1307,), (0.3081,)), params.not_number_train_size, params.not_number_test_size
    )
    run(train_dataset, test_dataset, params.batch_size, params.epochs, params.learning_rate, result_fig_path="imgs/not-number.png")
