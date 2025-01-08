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


import wandb

import csv
import os

if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'optimizer': {
                'values': ['sgd_nesterov', 'adamw']
            },
            'learning_rate': {
                'values': [0.01, 0.001, 0.0001]
            },
            'weight_decay': {
                'values': [0.001, 0.0001]
            },
            'scheduler': {
                'values': ['cosine', 'step', 'plateau']
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="noisy_cifar100")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results_file = "sweep_results.csv"

    if not os.path.exists(results_file):
        with open(results_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['run_id', 'optimizer', 'learning_rate', 'weight_decay', 'scheduler',
                             'best_val_accuracy', 'best_model_path'])

    EPOC = 10

    noise_file_path = 'data/CIFAR-100_human.pt'
    noise_data = torch.load(noise_file_path)
    noisy_labels = noise_data['noisy_label']

    train_transform = transforms.Compose([

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
    from tqdm import tqdm
    from torchvision.transforms import v2


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


    def train_model():
        wandb.init()
        config = wandb.config

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

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=True, num_workers=4,
                                  persistent_workers=True, prefetch_factor=2)
        test_loader = DataLoader(test_set, batch_size=256, pin_memory=True, num_workers=4, persistent_workers=True)

        model = create_model("resnet34", pretrained=True, num_classes=100).to(device)

        if config.optimizer == 'sgd_nesterov':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9,
                                        weight_decay=config.weight_decay, nesterov=True)
        elif config.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        if config.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOC)
        elif config.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        elif config.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        criterion = LabelSmoothingLoss(num_classes=100, smoothing=0.1)

        cutMix = v2.CutMix(num_classes=100, alpha=1.0)
        mixUp = v2.MixUp(num_classes=100, alpha=1.0)

        best_val_accuracy = 0.0
        best_model_path = None

        for epoch in range(EPOC):
            model.train()
            correct, total, total_loss = 0, 0, 0.0

            batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

            for inputs, targets in batch_bar:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

                if epoch < EPOC // 2:
                    inputs, targets = mixUp(inputs, targets)
                else:
                    inputs, targets = cutMix(inputs, targets)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                total += targets.size(0)

                if isinstance(targets, Tensor):
                    correct += (outputs.argmax(1) == targets.argmax(1)).sum().item()

                avg_loss = total_loss / total
                accuracy = 100.0 * correct / total

                batch_bar.set_postfix({
                    "Batch Loss": f"{loss.item():.4f}",
                    "Epoch Loss": f"{avg_loss:.4f}",
                    "Accuracy": f"{accuracy:.2f}%"
                })

            train_loss = avg_loss
            train_accuracy = accuracy

            model.eval()
            total_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                    outputs = model(inputs)
                    loss = functional.cross_entropy(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                    correct += (outputs.argmax(1) == targets).sum().item()
                    total += targets.size(0)

            val_loss = total_loss / total
            val_accuracy = 100.0 * correct / total

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            if config.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_path = f"best_model_{wandb.run.id}.pth"
                torch.save(model.state_dict(), best_model_path)

        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                wandb.run.id,
                config.optimizer,
                config.learning_rate,
                config.weight_decay,
                config.scheduler,
                best_val_accuracy,
                best_model_path
            ])

        wandb.finish()


    wandb.agent(sweep_id, function=train_model)
