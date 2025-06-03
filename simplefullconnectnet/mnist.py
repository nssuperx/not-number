import os
import sys
import torch
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simplefullconnectnet.simple_fullconnect_net import SimpleFullConnectNet, train, test


try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    device = torch.device("cpu")

if device.type == "cpu":
    try:
        import torch_directml  # noqa

        device = torch_directml.device()
    except ImportError:
        pass

if __name__ == "__main__":
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=100)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=100)

    model = SimpleFullConnectNet(28 * 28, 1000, len(train_dataset.classes)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(1, 10 + 1):
        train(model, train_loader, optimizer, epoch, device)
        test(model, test_loader, device)
