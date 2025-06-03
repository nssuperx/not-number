import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


# https://github.com/pytorch/examples/blob/main/mnist/main.py


class SimpleFullConnectNet(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, classes: int):
        super(SimpleFullConnectNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, classes)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = self.fc(x)
        x = self.out_layer(x)
        output = F.softmax(x, dim=1)
        return output


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: torch.device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output: torch.Tensor = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
