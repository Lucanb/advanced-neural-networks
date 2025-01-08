import psutil
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import torch.nn.functional as functional
from torchvision.datasets import CIFAR100
import pandas as pd
from torchvision import transforms
from torchvision.transforms import v2, AutoAugment, AutoAugmentPolicy, Resize
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def loss_coteaching(y1, y2, targets, forget_rate):
    loss_1 = functional.cross_entropy(y1, targets, reduction='none')
    loss_2 = functional.cross_entropy(y2, targets, reduction='none')

    ind_1_sorted = torch.argsort(loss_1)
    ind_2_sorted = torch.argsort(loss_2)

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    loss_update_1 = functional.cross_entropy(y1[ind_2_update], targets[ind_2_update])
    loss_update_2 = functional.cross_entropy(y2[ind_1_update], targets[ind_1_update])

    return loss_update_1, loss_update_2


def compute_pure_ratio(selected_indices, true_labels, noisy_labels):
    clean = true_labels[selected_indices] == noisy_labels[selected_indices]
    pure_ratio = clean.sum().item() / len(selected_indices)
    return pure_ratio

def get_memory_usage():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)
    return ram_usage

def train_one_epoch(epoch, model1, model2, optimizer1, optimizer2, train_loader, forget_rate):
    model1.train()
    model2.train()

    total_loss1, total_loss2, correct1, correct2, total = 0, 0, 0, 0, 0

    batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch_idx, (inputs, targets) in enumerate(batch_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        if batch_idx % 10 == 0:
            gpu_memory = torch.cuda.memory_allocated(device) / (1024 * 1024)
            cpu_memory = get_memory_usage()
            print(f"Batch {batch_idx}: GPU Memory: {gpu_memory:.2f} MB, CPU Memory: {cpu_memory:.2f} MB")

        outputs1 = model1(inputs)
        outputs2 = model2(inputs)

        loss1, loss2 = loss_coteaching(outputs1, outputs2, targets, forget_rate)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        total_loss1 += loss1.item() * inputs.size(0)
        total_loss2 += loss2.item() * inputs.size(0)
        total += inputs.size(0)

        predicted1 = outputs1.argmax(dim=1)
        predicted2 = outputs2.argmax(dim=1)

        correct1 += (predicted1 == targets).sum().item()
        correct2 += (predicted2 == targets).sum().item()

        avg_loss1 = total_loss1 / total
        avg_loss2 = total_loss2 / total
        acc1 = 100.0 * correct1 / total
        acc2 = 100.0 * correct2 / total

        batch_bar.set_postfix({
            "Loss1": f"{avg_loss1:.4f}",
            "Loss2": f"{avg_loss2:.4f}",
            "Acc1": f"{acc1:.2f}%",
            "Acc2": f"{acc2:.2f}%"
        })

    return avg_loss1, avg_loss2, acc1, acc2


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = functional.cross_entropy(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        predicted = outputs.argmax(dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return accuracy, avg_loss


def main():
    noise_file_path = 'data/CIFAR-100_human.pt'
    noise_data = torch.load(noise_file_path)
    noisy_labels = noise_data['noisy_label']

    transform = transforms.Compose([
        AutoAugment(AutoAugmentPolicy.CIFAR10),

        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_set = CIFAR100N(root='data/', transform=transform, noisy_labels=noisy_labels)
    test_set = CIFAR100(root='data/', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True, num_workers=4,
                              persistent_workers=True)

    test_loader = DataLoader(test_set, batch_size=128, pin_memory=True, num_workers=4, persistent_workers=True,
                             )

    initial_forget_rate = 0.4
    EPOCHS = 10

    model1 = create_model('resnet18', pretrained=True, num_classes=100).to(device)
    model2 = create_model('resnet18', pretrained=True, num_classes=100).to(device)

    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=0.001, weight_decay=0.001)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.001, weight_decay=0.001)

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=EPOCHS)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        forget_rate = initial_forget_rate * (1 - epoch / EPOCHS)

        train_loss1, train_loss2, train_acc1, train_acc2 = train_one_epoch(
            epoch, model1, model2, optimizer1, optimizer2, train_loader, forget_rate
        )

        val_acc1, val_loss1 = evaluate(model1, test_loader)
        val_acc2, val_loss2 = evaluate(model2, test_loader)

        scheduler1.step()
        scheduler2.step()

        print(f"Epoch {epoch + 1}:")
        print(f"Model 1 - Train Acc: {train_acc1:.2f}%, Val Acc: {val_acc1:.2f}%")
        print(f"Model 2 - Train Acc: {train_acc2:.2f}%, Val Acc: {val_acc2:.2f}%")


if __name__ == '__main__':
    main()
