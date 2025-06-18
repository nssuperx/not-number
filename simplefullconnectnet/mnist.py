from collections import Counter
import torch
import torch.optim as optim
from torchvision import datasets, transforms

from notnumberutil.util import get_device, show_accuracy_glaph
from simple_fullconnect_net import SimpleFullConnectNet, train, test

if __name__ == "__main__":
    device = get_device()
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=tf)
    test_dataset = datasets.MNIST("../data", train=False, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=100, pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=100, pin_memory=True, num_workers=4)

    test_classes = Counter(label for _, label in test_dataset)
    test_size = test_classes.total()

    model = SimpleFullConnectNet(28 * 28, [1000, 100, 500], len(test_classes.keys())).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    correct_classes_history: list[list[int]] = []
    for epoch in range(10):
        train(model, train_loader, optimizer, device)
        loss, correct = test(model, test_loader, len(test_classes.keys()), device)
        sc = sum(correct)
        correct_classes_history.append(correct)
        print(f"Epoch {epoch + 1:2d}, Test Loss: {loss:.4f}, Correct: {sc}/{test_size}, Accuracy: {sc / test_size * 100:.2f}%")

    last_correct = correct_classes_history[-1]
    class_acc = (f"{c}: {(last_correct[i] / test_classes[c] * 100):.2f}%" for i, c in enumerate(sorted(test_classes.keys())))
    print("Accuracy:", ", ".join(class_acc))
    show_accuracy_glaph(test_classes, correct_classes_history)
