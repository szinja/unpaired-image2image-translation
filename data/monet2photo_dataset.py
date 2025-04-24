import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob

class Monet2PhotoDataset(Dataset):
    def __init__(self, root_dir, domain="A", transform=None, mode="train"):
        self.transform = transform
        assert domain in ["A", "B"]
        assert mode in ["train", "test"]
        self.domain = domain
        subfolder = "trainA" if domain == "A" else "trainB"
        if mode == "test":
            subfolder = "testA" if domain == "A" else "testB"

        # CycleGAN property -> len(trainA + trainB)
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, subfolder, "*.jpg")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def get_dataloaders(root_dir, batch_size, resize, num_workers=2, pin_memory=True): # recommended num_workers is 2
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset_A = Monet2PhotoDataset(root_dir, domain="A", transform=transform, mode="train")
    dataset_B = Monet2PhotoDataset(root_dir, domain="B", transform=transform, mode="train")

    # Checking if datasets are not empty
    if len(dataset_A) == 0 or len(dataset_B) == 0:
         raise FileNotFoundError(f"Could not find images in trainA or trainB under {root_dir}")

    loader_A = DataLoader(
        dataset_A,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )
    loader_B = DataLoader(
        dataset_B,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True
    )
    return loader_A, loader_B

if __name__ == '__main__':
    root_dir = "data/monet2photo"
    batch_size = 1
    resize_shape = (256, 256)
    try:
        loader_monet, loader_photo = get_dataloaders(root_dir=root_dir, batch_size=batch_size, resize=resize_shape)

        # Example of iterating (use zip for training)
        print(f"Monet dataset size: {len(loader_monet.dataset)}")
        print(f"Photo dataset size: {len(loader_photo.dataset)}")

        # Fetch one batch from each for demonstration
        monet_batch = next(iter(loader_monet))
        photo_batch = next(iter(loader_photo))
        print(f"Fetched batch: Monet shape={monet_batch.shape}, Photo shape={photo_batch.shape}")

    except FileNotFoundError as e:
        print(e)
        print("Please ensure the monet2photo dataset is downloaded and extracted correctly.")
        print("Expected structure: data/monet2photo/trainA/*.jpg, data/monet2photo/trainB/*.jpg, etc.")