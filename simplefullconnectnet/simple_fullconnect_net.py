import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


# https://github.com/pytorch/examples/blob/main/mnist/main.py


class SimpleFullConnectNet(nn.Module):
    def __init__(self, in_size: int, hidden_sizes: list[int], classes: int):
        super(SimpleFullConnectNet, self).__init__()
        self.flatten = nn.Flatten()
        self.in_layer = nn.Linear(in_size, hidden_sizes[0])
        self.fcs = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes[:-1]))])
        self.out_layer = nn.Linear(hidden_sizes[-1], classes)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.in_layer(x)
        for fc in self.fcs:
            x = fc(x)
        x = self.out_layer(x)
        output = F.softmax(x, dim=1)
        return output


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


def test(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, int]:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output: torch.Tensor = model(data)
            test_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # pyright: ignore[reportArgumentType]
    return test_loss, int(correct)
