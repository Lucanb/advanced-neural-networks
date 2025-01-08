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

device = torch.device('cuda')
cudnn.benchmark = True
pin_memory = True
enable_half = True
scaler = GradScaler(device, enabled=enable_half)

print(device)
EPOC = 10


class CIFAR100N(Dataset):
    def __init__(self, root, transform=None, noise_file='./data/CIFAR-100_human.pt'):
        self.cifar100 = CIFAR100(root=root, train=True, download=True, transform=transform)
        noise_data = torch.load(noise_file)
        self.noisy_labels = noise_data['noisy_label']

    def __len__(self):
        return len(self.cifar100)

    def __getitem__(self, idx):
        image, _ = self.cifar100[idx]
        label = self.noisy_labels[idx]
        return image, label


def create_plots(f, train_accuracies, val_accuracies, train_losses, val_losses, epochs):
    if f is True:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_accuracies, label="Train Accuracy", marker='o')
        plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker='x')
        plt.title("Training and Validation Accuracy over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.legend(loc="best")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_losses, label="Train Loss", marker='o')
        plt.plot(epochs, val_losses, label="Validation Loss", marker='x')
        plt.title("Training and Validation Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.grid(True)

        plt.tight_layout()
        plt.show()


class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]



noise_file_path = '../data/CIFAR-100_human.pt'

train_set = CIFAR100N(
    root='/kaggle/input/fii-atnn-2024-assignment-2',
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ]),
    noise_file=noise_file_path
)

test_set = CIFAR100(
    root='/kaggle/input/fii-atnn-2024-assignment-2',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=256, pin_memory=True)


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Classifier
            nn.Flatten(),
            nn.Linear(512, 100)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


model = VGG16().to(device)
model = torch.jit.script(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2, nesterov=True, fused=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOC)

train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

cutMix = v2.CutMix(num_classes=100, alpha=1.0)
mixUp = v2.MixUp(num_classes=100, alpha=1.0)

rand_choice = v2.RandomChoice([cutMix, mixUp])


total_train_time = 0
total_val_time = 0
peak_memory_usage = 0

def train():
    global total_train_time, peak_memory_usage
    model.train()
    correct = 0
    total = 0
    total_loss = 0.0

    start_time = time.time()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        inputs, targets = rand_choice(inputs, targets)

        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        total_loss += loss.item() * inputs.size(0)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets.argmax(1)).sum().item()

        peak_memory_usage = max(peak_memory_usage, torch.cuda.max_memory_allocated(device))

    end_time = time.time()
    total_train_time += (end_time - start_time)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return accuracy, avg_loss


@torch.inference_mode()
def val():
    global total_val_time, peak_memory_usage
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    start_time = time.time()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=enable_half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        peak_memory_usage = max(peak_memory_usage, torch.cuda.max_memory_allocated(device))

    end_time = time.time()
    total_val_time += (end_time - start_time)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return accuracy, avg_loss

if __name__ == '__main__':
    best_model_state = model.state_dict()
    best = 0.0
    epochs = list(range(EPOC))
    with tqdm(epochs) as tbar:
        for epoch in tbar:
            train_acc, train_loss = train()

            train_accuracies.append(train_acc)
            train_losses.append(train_loss)

            val_acc, val_loss = val()
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)

            scheduler.step()

            if val_acc > best:
                best = val_acc
                torch.save(best_model_state, 'best_model.pth')
            tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.6f}, Best: {best:.6f}")

    create_plots(True, train_accuracies, val_accuracies, train_losses, val_losses, epochs)
    model.load_state_dict(torch.load('best_model.pth'))

    print(f"Total Training Time: {total_train_time:.2f} seconds")
    print(f"Total Validation Time: {total_val_time:.2f} seconds")
    print(f"Peak Memory Usage: {peak_memory_usage / (1024 ** 2):.2f} MB")

    data = {
        "ID": [],
        "target": []
    }

    for i, label in enumerate(inference()):
        data["ID"].append(i)
        data["target"].append(label)

    df = pd.DataFrame(data)
    df.to_csv(f"submission_new_{time.time()}.csv", index=False)

