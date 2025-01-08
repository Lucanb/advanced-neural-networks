import torch
from torch import nn, Tensor
import torch.nn.functional as functional
from torchvision.datasets import CIFAR100
import pandas as pd
from torchvision import transforms
from torchvision.transforms import v2, AutoAugment, AutoAugmentPolicy
from torch.backends import cudnn
from torch import GradScaler
from torch import optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from timm import create_model
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
cudnn.benchmark = True


class CachedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.data = [(image, label) for image, label in dataset]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.transform:
            image = self.transform(image)
        return image, label


class CIFAR100N(Dataset):
    def __init__(self, root, transform=None, noisy_labels=None):
        self.cifar100 = CIFAR100(root=root, train=True, download=True, transform=transform)
        self.noisy_labels = noisy_labels

    def __len__(self):
        return len(self.cifar100)

    def __getitem__(self, idx):
        image, _ = self.cifar100[idx]
        label = self.noisy_labels[idx]
        return image, label


def create_plots():
    epochs_range = list(range(1, EPOC + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(epochs_range, val_accuracies, label="Validation Accuracy", marker='x')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs_range, val_losses, label="Validation Loss", marker='x')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    noise_file_path = '../data/CIFAR-100_human.pt'
    noise_data = torch.load(noise_file_path)
    noisy_labels = noise_data['noisy_label']

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = CIFAR100N(
        root='/kaggle/input/fii-atnn-2024-assignment-2',
        transform=train_transform,
        noisy_labels=noisy_labels
    )

    test_set = CIFAR100(
        root='/kaggle/input/fii-atnn-2024-assignment-2',
        train=False,
        download=True,
        transform=test_transform
    )


    for i in range(5):
        image, label = train_set[i]
        print(f"Image shape after transform: {image.shape}, Label: {label}")

    assert len(train_set) == 50000, "Dataset length mismatch!"

    for i in range(5):
        image, noisy_label = train_set[i]
        clean_label = train_set.cifar100.targets[i]
        print(f"Index {i}: Clean Label: {clean_label}, Noisy Label: {noisy_label}")

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=True, num_workers=4,
                              persistent_workers=True, prefetch_factor=2)

    for batch_idx, (images, noisy_labels) in enumerate(train_loader):
        print(f"Batch {batch_idx} - First 5 Noisy Labels: {noisy_labels[:5]}")
        break

    test_loader = DataLoader(test_set, batch_size=256, pin_memory=True, num_workers=4, persistent_workers=True,
                             prefetch_factor=2)

    EPOC = 5
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class WideResNet(nn.Module):
        def __init__(self, depth, width, num_classes, dropout_rate=0.3):
            super(WideResNet, self).__init__()
            assert (depth - 4) % 6 == 0, "Depth must be 6n + 4"
            n = (depth - 4) // 6
            k = width

            self.in_planes = 16

            def conv3x3(in_planes, out_planes, stride=1):
                return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

            class BasicBlock(nn.Module):
                def __init__(self, in_planes, out_planes, stride, dropout_rate):
                    super(BasicBlock, self).__init__()
                    self.bn1 = nn.BatchNorm2d(in_planes)
                    self.conv1 = conv3x3(in_planes, out_planes, stride)
                    self.bn2 = nn.BatchNorm2d(out_planes)
                    self.conv2 = conv3x3(out_planes, out_planes, 1)
                    self.dropout_rate = dropout_rate

                    self.shortcut = nn.Sequential()
                    if stride != 1 or in_planes != out_planes:
                        self.shortcut = nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
                        )

                def forward(self, x):
                    out = F.relu(self.bn1(x))
                    out = self.conv1(out)
                    if self.dropout_rate > 0:
                        out = F.dropout(out, p=self.dropout_rate, training=self.training)
                    out = self.conv2(F.relu(self.bn2(out)))
                    out += self.shortcut(x)
                    return out

            class NetworkBlock(nn.Module):
                def __init__(self, num_layers, in_planes, out_planes, block, stride, dropout_rate):
                    super(NetworkBlock, self).__init__()
                    layers = []
                    for i in range(num_layers):
                        layers.append(
                            block(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1,
                                  dropout_rate)
                        )
                    self.layer = nn.Sequential(*layers)

                def forward(self, x):
                    return self.layer(x)

            self.conv1 = conv3x3(3, self.in_planes)
            self.layer1 = NetworkBlock(n, self.in_planes, 16 * k, BasicBlock, 1, dropout_rate)
            self.layer2 = NetworkBlock(n, 16 * k, 32 * k, BasicBlock, 2, dropout_rate)
            self.layer3 = NetworkBlock(n, 32 * k, 64 * k, BasicBlock, 2, dropout_rate)
            self.bn1 = nn.BatchNorm2d(64 * k)
            self.fc = nn.Linear(64 * k, num_classes)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.size(3))
            out = out.view(out.size(0), -1)
            return self.fc(out)


    # model = WideResNet(depth=28, width=2, num_classes=100)

    model = create_model("resnet18", pretrained=True, num_classes=100)
    print(model.pretrained_cfg)

    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOC)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOC)

    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    cutMix = v2.CutMix(num_classes=100, alpha=1.0)
    mixUp = v2.MixUp(num_classes=100, alpha=1.0)

    rand_choice = v2.RandomChoice([cutMix, mixUp])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    total_train_time = 0
    total_eval_time = 0
    peak_memory_usage = 0


    def train_one_epoch(epoch):
        global total_train_time, peak_memory_usage
        model.train()
        correct, total, total_loss = 0, 0, 0.0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        start_time = time.time()

        for inputs, targets in batch_bar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            inputs, targets = rand_choice(inputs, targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets.argmax(1)).sum().item()

            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total
            batch_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Epoch Loss": f"{avg_loss:.4f}",
                "Accuracy": f"{accuracy:.2f}%"
            })

            peak_memory_usage = max(peak_memory_usage, torch.cuda.max_memory_allocated(device))

        end_time = time.time()
        total_train_time += (end_time - start_time)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return accuracy, avg_loss


    @torch.no_grad()
    def evaluate():
        global total_eval_time, peak_memory_usage
        model.eval()
        correct, total, total_loss = 0, 0, 0.0

        start_time = time.time()

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            peak_memory_usage = max(peak_memory_usage, torch.cuda.max_memory_allocated(device))

        end_time = time.time()
        total_eval_time += (end_time - start_time)

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return accuracy, avg_loss


    best_model_state = model.state_dict()
    best_val_accuracy = 0.0

    print("Starting training...")
    for epoch in range(EPOC):
        train_acc, train_loss = train_one_epoch(epoch)
        val_acc, val_loss = evaluate()

        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        scheduler.step()

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict()

        print(
            f"Epoch {epoch + 1}/{EPOC} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Best Val Acc: {best_val_accuracy:.2f}%")
        print()

    torch.save(best_model_state, "../models/best_model.pth")

    print(f"Total Training Time: {total_train_time:.2f} seconds")
    print(f"Total Evaluation Time: {total_eval_time:.2f} seconds")
    print(f"Peak Memory Usage: {peak_memory_usage / (1024 ** 2):.2f} MB")

    create_plots()
