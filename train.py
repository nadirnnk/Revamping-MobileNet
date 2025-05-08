import torch
import torch.nn as nn
import torch.optim as optim
# from model import CelebAMaskLiteUNet
# from model_2 import MobileNetV3_Segmenter
# from dataset import CelebAMaskDataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.data import random_split
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)  # Convert to probabilities
        targets = F.one_hot(targets, num_classes=19).permute(0, 3, 1, 2).float()  # One-hot encode
        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()  # Loss is 1 - Dice coefficient

# Combined loss
class CombinedLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss

# Update criterion
# alpha = torch.tensor([0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
criterion = CombinedLoss()


transforms_val = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Convert to tensor
    ])



train_dataset = CelebAMaskDataset(image_dir, mask_dir, transform=transforms_val, tag='train')
val_dataset = CelebAMaskDataset(image_val_dir, mask_val_dir, transform=transforms_val, tag='val')

# Step 6: Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)


train_list=[]
val_list=[]
epochs_list=[]


# Define the model (assuming CelebAMaskLiteUNet is already defined)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# model = CelebAMaskLiteUNet(base_channels=30, num_classes=19).to(device)
model = MobileNetV3_Segmenter(num_classes=19).to(device)

# Define loss function and optimizer
criterion = CombinedLoss(gamma=2.0, dice_weight=0.5) # For multi-class segmentation
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)


# Early stopping params
early_stopping_patience = 10   # stop if no improvement after this many epochs
best_val_loss = float('inf')
epochs_no_improve = 0

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()
    val_loss /= len(val_loader)

    # Step the scheduler
    scheduler.step(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] — "
          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} — "
          f"LR: {scheduler._last_lr[0]:.2e}")

    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # save the best model
        torch.save(model.state_dict(), 'best_ckpt.pth')
        print(f"  ➜ New best validation loss. Saving model.")
    else:
        epochs_no_improve += 1
        print(f"  ➜ No improvement for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# at the end, you can load the best model if you like:
model.load_state_dict(torch.load('ckpt_reg.pth'))
# torch.save(model.state_dict(), 'ckpt.pth')  # optionally save final version