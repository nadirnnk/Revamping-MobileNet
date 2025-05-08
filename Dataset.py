# dataset.py
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


class CelebAMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, tag=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.tag = tag
        # self.trns = trns


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask

        transforms_train = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),  # Horizontal flip
        v2.RandomRotation(degrees=15),  # Rotation ±15°
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),  # Color jitter
        # v2.RandomResizedCrop(size=(256, 256), scale=(0.9, 1.1)),  # Small scale variations
        ])

        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # if self.tag=='train':
        #     image, mask = transforms_train(image, mask)

        # Convert mask to tensor (class indices, not one-hot encoding)
        mask = torch.tensor(np.array(mask), dtype=torch.long)


        if self.transform:
            image = self.transform(image)

        return image, mask