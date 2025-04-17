from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MonetPhotoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_A = os.path.join(root_dir, "trainA")
        self.root_B = os.path.join(root_dir, "trainB")
        self.images_A = os.listdir(self.root_A)
        self.images_B = os.listdir(self.root_B)
        self.length_dataset = max(len(self.images_A), len(self.images_B)) # CycleGAN property
        self.transform = transform

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        img_A = Image.open(os.path.join(self.root_A, self.images_A[idx % len(self.images_A)]))
        img_B = Image.open(os.path.join(self.root_B, self.images_B[idx % len(self.images_B)]))

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B

def get_loaders(
    root_dir,
    batch_size,
    resize,
    num_workers=4,
    pin_memory=True,
):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize to [-1, 1]
    ])

    dataset = MonetPhotoDataset(root_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    return loader

if __name__ == '__main__':
    # Example Usage
    root_dir = "data/monet2photo"  # Replace with your dataset path
    batch_size = 1
    resize = 256
    loader = get_loaders(root_dir=root_dir, batch_size=batch_size, resize=(resize, resize))

    for i, (monet, photo) in enumerate(loader):
        print(f"Batch {i+1}: Monet shape={monet.shape}, Photo shape={photo.shape}")
        break