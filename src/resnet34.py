import random

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
        self.noisy_labels = torch.tensor(noisy_labels, dtype=torch.long)

    def __len__(self):
        return len(self.cifar100)

    def __getitem__(self, idx):
        image, _ = self.cifar100[idx]
        label = self.noisy_labels[idx]
        return image, label

    def check_noise_percentage(self):
        """
        Check the percentage of noisy labels in the dataset.
        """
        clean_labels = torch.tensor(self.cifar100.targets)
        noisy_labels = torch.tensor(self.noisy_labels)
        assert len(clean_labels) == len(noisy_labels), "Mismatch in label lengths"

        mismatches = (clean_labels != noisy_labels).sum().item()
        total_labels = len(clean_labels)

        noise_percentage = (mismatches / total_labels) * 100
        print(f"Noise Percentage: {noise_percentage:.2f}%")
        return noise_percentage


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
    noise_file_path = 'data/CIFAR-100_human.pt'
    noise_data = torch.load(noise_file_path)
    noisy_labels = noise_data['noisy_label']

    train_transform = transforms.Compose([
        AutoAugment(AutoAugmentPolicy.CIFAR10),

        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])


    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = CIFAR100N(
        root='/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100',
        transform=train_transform,
        noisy_labels=noisy_labels
    )

    test_set = CIFAR100(
        root='/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100',
        train=False,
        download=True,
        transform=test_transform
    )

    noise_percentage = train_set.check_noise_percentage()
    print(f"The dataset contains {noise_percentage:.2f}% noisy labels.")

    assert len(train_set) == 50000, "Dataset length mismatch!"

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True, num_workers=4,
                              persistent_workers=True, prefetch_factor=2)

    test_loader = DataLoader(test_set, batch_size=128, pin_memory=True, num_workers=4, persistent_workers=True,
                             prefetch_factor=2)

    EPOC = 20
    import torch
    import torch.nn as nn

    model = create_model("resnet34", pretrained=True, num_classes=100)

    print(model.pretrained_cfg)

    model = model.to('cuda')


    class LabelSmoothingLoss(nn.Module):
        def __init__(self, num_classes, smoothing=0.1):
            super(LabelSmoothingLoss, self).__init__()
            self.smoothing = smoothing
            self.num_classes = num_classes

        def forward(self, preds, targets):
            if targets.dtype == torch.float:
                smooth_labels = targets
            else:
                one_hot = torch.zeros_like(preds).scatter(1, targets.unsqueeze(1), 1)
                smooth_labels = (1 - self.smoothing) * one_hot + self.smoothing / self.num_classes

            log_probs = torch.log_softmax(preds, dim=1)
            return -torch.mean(torch.sum(smooth_labels * log_probs, dim=1))


    criterion = LabelSmoothingLoss(num_classes=100, smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOC)


    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    cutMix = v2.CutMix(num_classes=100, alpha=1.0)
    mixUp = v2.MixUp(num_classes=100, alpha=1.0)

    rand_choice = v2.RandomChoice([cutMix, mixUp])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    enable_half = True  # Disable for CPU, it is slower!


    def train_one_epoch(epoch):
        model.train()
        correct, total, total_loss = 0, 0, 0.0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for inputs, targets in batch_bar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            if epoch < EPOC // 2:
                inputs, targets = mixUp(inputs, targets)
            else:
                inputs, targets = cutMix(inputs, targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(1)
            total += targets.size(0)

            if isinstance(targets, Tensor):
                correct += (predicted == targets.argmax(1)).sum().item()

            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total
            batch_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "Epoch Loss": f"{avg_loss:.4f}",
                "Accuracy": f"{accuracy:.2f}%"
            })

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return accuracy, avg_loss


    @torch.no_grad()
    def evaluate():
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = functional.cross_entropy(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return accuracy, avg_loss


    @torch.inference_mode()
    def inference():
        model.eval()

        labels = []

        for inputs, _ in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=enable_half):
                outputs = model(inputs)

            predicted = outputs.argmax(1).tolist()
            labels.extend(predicted)

        return labels


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
            f"\nEpoch {epoch + 1}/{EPOC} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Best Val Acc: {best_val_accuracy:.2f}%")
        print()

    torch.save(best_model_state, f"best_model_here.pth")
    create_plots()

    data = {
        "ID": [],
        "target": []
    }

    for i, label in enumerate(inference()):
        data["ID"].append(i)
        data["target"].append(label)

    df = pd.DataFrame(data)
    df.to_csv("submission.csv", index=False)
