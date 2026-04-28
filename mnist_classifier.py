"""
mnist_classifier.py
-----------------------
A simple MNIST classifier using PyTorch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from visualize import (plot_training_curves,
                       plot_confusion_matrix,
                       plot_predictions,
                       plot_class_accuracy,
                       plot_mistakes,
                       plot_dashboard,
                       EpochRecord)

logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_SYNC_FREQ = 100
EPOCH = 2
VIS_SAVE_DIR = None


class Dataset:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            (0.1307,), (0.3081,)
            )
        ])
        self.train_dataset = datasets.MNIST(root='./data', train=True,
                                            download=True, transform=self.transform)
        self.test_dataset = datasets.MNIST(root='./data', train=False,
                                           download=True, transform=self.transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32*14*14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
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
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def train_one_epoch(self, device, train_loader, optimizer, epoch):
        self.train()
        total_loss = 0.0
        total_samples = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if batch_idx % 100 == 0:
                logging.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)})'
                             f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        return total_loss / max(total_samples, 1)
                
    def evaluate(self, device, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                prediction = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += prediction.eq(target.view_as(prediction)).sum().item()
            
            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
            logging.info(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
                         f' ({accuracy:.0f}%)\n')
        return test_loss, accuracy

    def plot_predictions(self, device, test_loader):
        self.eval()
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = self(data)
        predictions = output.argmax(dim=1, keepdim=True).cpu().numpy()
        data = data.cpu().numpy()
        target = target.cpu().numpy()

        fig, axes = plt.subplots(1, 10, figsize=(15, 3))
        for i in range(10):
            axes[i].imshow(data[i][0], cmap='gray')
            axes[i].set_title(f'Pred: {predictions[i][0]}, True: {target[i]}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    dataset = Dataset(batch_size=BATCH_SIZE)
    history = []
    save_dir = Path(VIS_SAVE_DIR) if VIS_SAVE_DIR else None

    for epoch in range(1, EPOCH + 1):
        train_loss = model.train_one_epoch(device, dataset.train_loader, optimizer, epoch)
        val_loss, val_acc = model.evaluate(device, dataset.test_loader)
        history.append(EpochRecord(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_acc=val_acc))

    plot_training_curves(history, save_path=save_dir)
    plot_predictions(model, dataset.test_loader, device, save_path=save_dir)
    plot_class_accuracy(model, dataset.test_loader, device, save_path=save_dir)
    plot_confusion_matrix(model, dataset.test_loader, device, save_path=save_dir)
    plot_mistakes(model, dataset.test_loader, device, save_path=save_dir)
    plot_dashboard(model, dataset.test_loader, device, history, save_path=save_dir)
        
    
if __name__ == '__main__':
    main()