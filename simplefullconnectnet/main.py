import os
import sys
import torch
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.concat_mnist import generate_mnist_with_fmnist
from simplefullconnectnet.simple_fullconnect_net import SimpleFullConnectNet, train, test

if __name__ == "__main__":
    train_dataset, test_dataset = generate_mnist_with_fmnist()
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=64)

    model = SimpleFullConnectNet(28 * 28, 100, 11)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, 100 + 1):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
