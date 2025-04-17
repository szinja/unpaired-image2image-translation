from torch.utils.data import Dataset
import cv2
import os
import torch
import numpy as np

class Monet2PhotoDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.images = image_paths
        self.masks = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_GRAYSCALE) / 255.0
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE) / 255.0

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
        }