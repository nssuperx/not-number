import os
import sys
import torch
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from simplefullconnectnet.simple_fullconnect_net import SimpleFullConnectNet, train, test
from util import get_device

if __name__ == "__main__":
    device = get_device()
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST("../data", train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=100, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=100, pin_memory=True, num_workers=4)

    model = SimpleFullConnectNet(28 * 28, [1000, 100, 500], len(train_dataset.classes)).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(1, 10 + 1):
        train(model, train_loader, optimizer, epoch, device)
        test(model, test_loader, device)
