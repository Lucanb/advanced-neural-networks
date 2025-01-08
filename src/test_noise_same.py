import torch
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
import psutil
import gc


class CIFAR100_noisy_fine(Dataset):
    def __init__(
            self, root: str, train: bool, transform: callable, download: bool
    ):
        cifar100 = CIFAR100(
            root=root, train=train, transform=transform, download=download
        )
        data, targets = tuple(zip(*cifar100))

        if train:
            noisy_label_file = "CIFAR-100-noisy.npz"
            if not os.path.isfile(noisy_label_file):
                raise FileNotFoundError(
                    f"{type(self).__name__} needs {noisy_label_file} to be used!"
                )

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file["clean_label"], targets):
                raise RuntimeError("Clean labels do not match!")
            targets = noise_file["noisy_label"]

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i]


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
        noisy_labels = self.noisy_labels
        assert len(clean_labels) == len(noisy_labels), "Mismatch in label lengths"

        mismatches = (clean_labels != noisy_labels).sum().item()
        total_labels = len(clean_labels)

        noise_percentage = (mismatches / total_labels) * 100
        print(f"Noise Percentage: {noise_percentage:.2f}%")
        return noise_percentage


def get_ram_usage():
    """Returns the current RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def check_datasets(train_set_1, train_set_2, clean_labels):
    assert len(train_set_1) == len(train_set_2), "Dataset lengths do not match!"

    noisy_labels_1 = torch.tensor([train_set_1[i][1] for i in range(len(train_set_1))])
    noisy_labels_2 = torch.tensor([train_set_2[i][1] for i in range(len(train_set_2))])

    if torch.equal(noisy_labels_1, noisy_labels_2):
        print("Noisy labels match perfectly!")
    else:
        print("Noisy labels do not match!")

    noise_percentage_1 = (noisy_labels_1 != clean_labels).sum().item() / len(train_set_1) * 100
    noise_percentage_2 = (noisy_labels_2 != clean_labels).sum().item() / len(train_set_2) * 100

    if abs(noise_percentage_1 - noise_percentage_2) < 1e-5:
        print(f"Noise percentages match: {noise_percentage_1:.2f}%")
    else:
        print(f"Noise percentages do not match: {noise_percentage_1:.2f}% vs {noise_percentage_2:.2f}%")

    for i in range(len(train_set_1)):
        img1, label1 = train_set_1[i]
        img2, label2 = train_set_2[i]

        if not torch.equal(img1, img2):
            print(f"Images do not match at index {i}!")
            return
        if label1 != label2:
            print(f"Labels do not match at index {i}!")
            return

    print("All samples and labels match perfectly!")


if __name__ == '__main__':
    print(f"Initial RAM usage: {get_ram_usage():.2f} MB")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    clean_dataset = CIFAR100(
        root='/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100',
        train=True, download=True
    )
    clean_labels = torch.tensor(clean_dataset.targets)

    noise_file_path = 'data/CIFAR-100_human.pt'
    noise_data = torch.load(noise_file_path)
    noisy_labels = noise_data['noisy_label']

    print(f"RAM after loading clean CIFAR-100: {get_ram_usage():.2f} MB")

    train_set_1 = CIFAR100_noisy_fine(
        root='/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100',
        download=True, train=True, transform=train_transform
    )
    print(f"RAM after loading CIFAR100_noisy_fine: {get_ram_usage():.2f} MB")

    assert len(clean_labels) == len(noisy_labels), \
        "Mismatch between the number of clean and noisy labels!"

    train_set_2 = CIFAR100N(
        root='/kaggle/input/fii-atnn-2024-project-noisy-cifar-100/fii-atnn-2024-project-noisy-cifar-100',
        transform=train_transform, noisy_labels=noisy_labels
    )
    print(f"RAM after loading CIFAR100N: {get_ram_usage():.2f} MB")

    check_datasets(train_set_1, train_set_2, clean_labels)

    del train_set_1, train_set_2, clean_dataset, clean_labels, noisy_labels
    gc.collect()

    torch.cuda.empty_cache()
    import gc

    print("Live objects in memory:")
    for obj in gc.get_objects():
        if isinstance(obj, Dataset):
            print(type(obj), obj)

    print(f"RAM after forced cleanup: {get_ram_usage():.2f} MB")
