import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision.transforms as tr
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class ImageDatasetNoLabel(Dataset):
    def __init__(self, class_path, transforms=None):
        super().__init__()
        self.class_label = class_path.split('/')[-1]
        self.img_paths = [os.path.join(class_path, img_name) for img_name in os.listdir(class_path)]
        self.transforms = transforms

    def __getitem__(self, index):
        img = cv2.cvtColor(cv2.imread(self.img_paths[index]), cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.img_paths)


@dataclass
class DatasetsClass:
    train_a: ImageDatasetNoLabel
    train_b: ImageDatasetNoLabel
    test_a: ImageDatasetNoLabel
    test_b: ImageDatasetNoLabel


@dataclass
class DataLoadersClass:
    train_a: DataLoader
    train_b: DataLoader
    test_a: DataLoader
    test_b: DataLoader


def get_channel_statistics(dataset):
    channel_sum = None
    channel_sq_sum = None
    pixel_count = 0

    for img in dataset:
        img = img / 255.0
        h, w, c = img.shape
        pixel_count += h * w

        if channel_sum is None:
            channel_sum = np.zeros(c, dtype=np.float32)
            channel_sq_sum = np.zeros(c, dtype=np.float32)

        channel_sum += img.sum(axis=(0, 1))
        channel_sq_sum += (img * img).sum(axis=(0, 1))

    channel_mean = channel_sum / pixel_count
    channel_std = np.sqrt(np.maximum(channel_sq_sum / pixel_count - channel_mean ** 2, 0))
    return channel_mean, channel_std


def get_transforms(channel_mean, channel_std, upscale_size=286, crop_size=256):
    train_transform = tr.Compose([
        tr.ToPILImage(),
        tr.Resize((upscale_size, upscale_size)),
        tr.RandomCrop(crop_size),
        tr.RandomHorizontalFlip(p=0.5),
        tr.ToTensor(),
        tr.Normalize(channel_mean, channel_std),
    ])

    val_transform = tr.Compose([
        tr.ToPILImage(),
        tr.Resize((crop_size, crop_size)),
        tr.ToTensor(),
        tr.Normalize(channel_mean, channel_std),
    ])

    def de_normalize(image):
        mean = torch.tensor(channel_mean, device=image.device).view(3, 1, 1)
        std = torch.tensor(channel_std, device=image.device).view(3, 1, 1)
        return (image * std + mean).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()

    return train_transform, val_transform, de_normalize


def show_examples(dataset, transform, de_norm, num_per_image=3, image_index=0, title=""):
    fig, ax = plt.subplots(1, 1 + num_per_image, figsize=(5 * (1 + num_per_image), 5))
    image = dataset[image_index]

    plt.suptitle(title, y=0.95)
    plt.subplot(1, 1 + num_per_image, 1)
    plt.imshow(image)
    plt.title("original")

    for i in range(num_per_image):
        plt.subplot(1, 1 + num_per_image, i + 2)
        plt.title(f"#{i}")
        plt.imshow(de_norm(transform(image)))
    plt.show()


def build_datasets(
    target_folder,
    train_transform_a,
    val_transform_a,
    train_transform_b,
    val_transform_b,
):
    return DatasetsClass(
        train_a=ImageDatasetNoLabel(os.path.join(target_folder, "trainA"), transforms=train_transform_a),
        train_b=ImageDatasetNoLabel(os.path.join(target_folder, "trainB"), transforms=train_transform_b),
        test_a=ImageDatasetNoLabel(os.path.join(target_folder, "testA"), transforms=val_transform_a),
        test_b=ImageDatasetNoLabel(os.path.join(target_folder, "testB"), transforms=val_transform_b),
    )


def build_dataloaders(ds: DatasetsClass, batch_size=1, num_workers=4, pin_memory=True):
    common = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    return DataLoadersClass(
        train_a=DataLoader(ds.train_a, batch_size=batch_size, shuffle=True, drop_last=True, **common),
        train_b=DataLoader(ds.train_b, batch_size=batch_size, shuffle=True, drop_last=True, **common),
        test_a=DataLoader(ds.test_a, batch_size=batch_size, shuffle=False, drop_last=True, **common),
        test_b=DataLoader(ds.test_b, batch_size=batch_size, shuffle=False, drop_last=True, **common),
    )
