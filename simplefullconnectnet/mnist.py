import os
import sys
import torch
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simplefullconnectnet.simple_fullconnect_net import SimpleFullConnectNet, train, test

if __name__ == "__main__":
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=64)

    model = SimpleFullConnectNet(28 * 28, 100, len(train_dataset.classes))
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(1, 10 + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
