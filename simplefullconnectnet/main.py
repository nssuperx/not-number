import os
import sys
import torch
import torch.optim as optim
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset.concat_mnist import generate_mnist_with_fmnist
from simplefullconnectnet.simple_fullconnect_net import SimpleFullConnectNet, train, test
from util import get_device

if __name__ == "__main__":
    device = get_device()
    train_dataset, test_dataset = generate_mnist_with_fmnist(transform=transforms.Normalize((0.1307,), (0.3081,)))
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=100, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=100, pin_memory=True, num_workers=4)
    classes = 10 + 1
    model = SimpleFullConnectNet(28 * 28, [1000, 100, 500], classes).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(10):
        train_loss = train(model, train_loader, optimizer, device)
        loss, correct = test(model, test_loader, device)
        print(
            f"Epoch {epoch + 1:2d}, Test Loss: {loss:.4f}, Correct: {correct}/{len(test_dataset)}, Accuracy: {correct / len(test_dataset) * 100:.2f}%"
        )
