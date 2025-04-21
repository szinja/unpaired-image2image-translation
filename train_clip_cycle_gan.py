import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import argparse
import wandb
from models.generator import Generator
from losses.modified_cycle_gan_loss import ModifiedCycleGANLoss
from data.monet2photo_dataset import Monet2PhotoDataset

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        wandb.init(
            project="CLIP_CycleGAN",
            entity="szinja-university-of-rochester"
        )
        wandb.config.update({
            "learning_rate": 0.0002,
            "batch_size": 64
        })
    except Exception as e:
        print(f"Could not initialize Wandb: {e}. Training without logging.")
        wandb.init(mode="disabled") # in case we train without Wandb

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # normalize to [-1, 1] first
    ])

    # Load datasets
    try:
        dataset_A = Monet2PhotoDataset(root_dir=args.dataset_path, domain="A", transform=transform, mode="train")
        dataset_B = Monet2PhotoDataset(root_dir=args.dataset_path, domain="B", transform=transform, mode="train")
        loader_A = DataLoader(dataset_A, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        loader_B = DataLoader(dataset_B, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        print(f"Dataset A ({dataset_A.domain}) size: {len(dataset_A)}")
        print(f"Dataset B ({dataset_B.domain}) size: {len(dataset_B)}")
        # datasets not empty
        if len(dataset_A) == 0 or len(dataset_B) == 0:
             raise FileNotFoundError(f"Could not find images in trainA or trainB under {args.dataset_path}")

    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset path is correct and contains trainA/trainB folders.")
        wandb.finish()
        return
    except Exception as e:
        print(f"An unexpected error occurred during dataset loading: {e}")
        wandb.finish()
        return

    G_A2B = Generator(n_residual_blocks=args.n_res_blocks).to(device)
    G_B2A = Generator(n_residual_blocks=args.n_res_blocks).to(device)

    # Local checkpoint
    if args.load_epoch > 0:
        try:
            print(f"Loading models from epoch {args.load_epoch}...")
            G_A2B.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"G_A2B_epoch_{args.load_epoch}.pth"), map_location=device))
            G_B2A.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"G_B2A_epoch_{args.load_epoch}.pth"), map_location=device))
            print("Models loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: Checkpoint files for epoch {args.load_epoch} not found. Starting from scratch.")
            args.load_epoch = 0 # reset load_epoch if we cannot find the files
        except Exception as e:
            print(f"Error loading checkpoints: {e}. Starting from scratch.")
            args.load_epoch = 0


    loss_fn = ModifiedCycleGANLoss(
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        lambda_clip=args.lambda_clip
    ).to(device)

    optimizer_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Starting training from epoch {args.load_epoch + 1}...")

    # Training loop
    start_epoch = args.load_epoch + 1
    for epoch in range(start_epoch, args.epochs + 1):
        G_A2B.train()
        G_B2A.train()

        epoch_G_loss = 0.0

        num_batches = min(len(loader_A), len(loader_B)) # run through the minimum length

        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            real_A, real_B = real_A.to(device), real_B.to(device)

            # Generator training
            optimizer_G.zero_grad()

            # Forward pass
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            cycled_A = G_B2A(fake_B)
            cycled_B = G_A2B(fake_A)
            identity_A = G_B2A(real_A) 
            identity_B = G_A2B(real_B)

            # Discriminator set to None - CLIP takes over
            disc_fake_A = None 
            disc_fake_B = None

            _, G_loss = loss_fn( # D_loss is ignored here as we don't train discriminators
                real_A=real_A, fake_B=fake_B, real_B=real_B, fake_A=fake_A,
                cycled_A=cycled_A, cycled_B=cycled_B,
                identity_A=identity_A if args.lambda_identity > 0 else None,
                identity_B=identity_B if args.lambda_identity > 0 else None,
                target_text_A=["a painting by Monet"] * real_A.size(0), # example prompt for domain A (Monet)
                target_text_B=["a photo of a landscape"] * real_B.size(0),   # example prompt for domain B (Photo)
                disc_fake_A=disc_fake_A,
                disc_fake_B=disc_fake_B
            )

            G_loss.backward()
            optimizer_G.step()

            # Logging
            epoch_G_loss += G_loss.item()
            current_D_loss = 0

            if (i + 1) % args.log_interval == 0:
                steps_done = (epoch - start_epoch) * num_batches + (i + 1)
                avg_G_loss_batch = epoch_G_loss / (i + 1)

                print(f"Epoch [{epoch}/{args.epochs}] Batch [{i+1}/{num_batches}] G_loss: {G_loss.item():.4f} (Avg: {avg_G_loss_batch:.4f})")

                # Log to Wandb
                log_data = {"G_loss": G_loss.item()}
                wandb.log(log_data, step=steps_done)

        avg_G_loss_epoch = epoch_G_loss / num_batches

        print(f"--- Epoch {epoch} Finished --- Avg G_loss: {avg_G_loss_epoch:.4f}")
        epoch_log_data = {"Epoch G_loss": avg_G_loss_epoch, "Epoch": epoch}
        wandb.log(epoch_log_data, step=(epoch - start_epoch + 1) * num_batches)

        # Test around 5 sample images
        if epoch % args.sample_interval == 0:
            with torch.no_grad():
                G_A2B.eval()
                G_B2A.eval()
                sample_real_A = next(iter(loader_A))[0:min(args.batch_size, 5)].to(device)
                sample_real_B = next(iter(loader_B))[0:min(args.batch_size, 5)].to(device)
                sample_fake_B = G_A2B(sample_real_A)
                sample_fake_A = G_B2A(sample_real_B)

                # Unnormalize images for logging if needed (assuming [-1, 1] range)
                sample_real_A = (sample_real_A * 0.5) + 0.5
                sample_real_B = (sample_real_B * 0.5) + 0.5
                sample_fake_A = (sample_fake_A * 0.5) + 0.5
                sample_fake_B = (sample_fake_B * 0.5) + 0.5

                wandb.log({
                    "Real A": [wandb.Image(img) for img in sample_real_A],
                    "Fake B (A->B)": [wandb.Image(img) for img in sample_fake_B],
                    "Real B": [wandb.Image(img) for img in sample_real_B],
                    "Fake A (B->A)": [wandb.Image(img) for img in sample_fake_A]
                }, step=(epoch - start_epoch + 1) * num_batches)


        # Another checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            print(f"Saving models at epoch {epoch}...")
            torch.save(G_A2B.state_dict(), os.path.join(args.checkpoint_dir, f"G_A2B_epoch_{epoch}.pth"))
            torch.save(G_B2A.state_dict(), os.path.join(args.checkpoint_dir, f"G_B2A_epoch_{epoch}.pth"))
            print("Models saved.")

    print("Training finished.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP-Enhanced CycleGAN")

    # Paths and logging
    parser.add_argument('--dataset_path', type=str, required=True, help='Root directory of the monet2photo dataset (containing trainA, trainB, etc.)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--wandb_project', type=str, default='clip-cyclegan-monet', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity (username or team name)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for Adam optimizer')
    parser.add_argument('--img_size', type=int, default=256, help='Image size to resize to')
    parser.add_argument('--n_res_blocks', type=int, default=9, help='Number of residual blocks in generators')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader worker processes')

    # Loss weights
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Weight for cycle consistency loss')
    parser.add_argument('--lambda_identity', type=float, default=0.5, help='Weight for identity loss (set to 0 to disable)')
    parser.add_argument('--lambda_clip', type=float, default=1.0, help='Weight for CLIP semantic loss')

    # Logging
    parser.add_argument('--log_interval', type=int, default=100, help='Log training status every N batches')
    parser.add_argument('--sample_interval', type=int, default=1, help='Save sample images to Wandb every N epochs')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Save model checkpoints every N epochs')

    # Resuming training
    parser.add_argument('--load_epoch', type=int, default=0, help='Epoch number to load checkpoint from (0 for starting from scratch)')

    args = parser.parse_args()
    print("Arguments:", args)
    train(args)