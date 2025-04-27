import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
from models.generator import Generator
from models.discriminator import Discriminator
from data.monet2photo_dataset import Monet2PhotoDataset

# Training a simple baseline CycleGAN to compare our CLIP-enhanced model
def train(args):
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    dataset_A = Monet2PhotoDataset(args.dataset_path, domain="A", mode="train", transform=transform)
    dataset_B = Monet2PhotoDataset(args.dataset_path, domain="B", mode="train", transform=transform)

    loader_A = DataLoader(dataset_A, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader_B = DataLoader(dataset_B, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    g_optimizer = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )
    d_optimizer = torch.optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )

    for epoch in range(1, args.epochs + 1):
        for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Training our generators
            g_optimizer.zero_grad()

            fake_B = G_A2B(real_A)
            recov_A = G_B2A(fake_B)
            fake_A = G_B2A(real_B)
            recov_B = G_A2B(fake_A)

            idt_A = G_B2A(real_A)
            idt_B = G_A2B(real_B)
            loss_idt_A = criterion_identity(idt_A, real_A) * args.lambda_identity
            loss_idt_B = criterion_identity(idt_B, real_B) * args.lambda_identity

            pred_fake_B = D_B(fake_B)
            pred_fake_A = D_A(fake_A)
            loss_gan_A2B = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
            loss_gan_B2A = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))

            loss_cycle_A = criterion_cycle(recov_A, real_A) * args.lambda_cycle
            loss_cycle_B = criterion_cycle(recov_B, real_B) * args.lambda_cycle

            g_loss = loss_gan_A2B + loss_gan_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            g_loss.backward()
            g_optimizer.step()

            # Training our discriminators
            d_optimizer.zero_grad()

            pred_real_A = D_A(real_A)
            pred_real_B = D_B(real_B)
            loss_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
            loss_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))

            pred_fake_A = D_A(fake_A.detach())
            pred_fake_B = D_B(fake_B.detach())
            loss_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
            loss_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))

            loss_D_A = (loss_real_A + loss_fake_A) * 0.5
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5

            d_loss = loss_D_A + loss_D_B
            d_loss.backward()
            d_optimizer.step()

            # wandb logging
            if i % args.log_interval == 0:
                wandb.log({
                    "Generator Loss": g_loss.item(),
                    "Discriminator Loss": d_loss.item(),
                    "Cycle Loss A": loss_cycle_A.item(),
                    "Cycle Loss B": loss_cycle_B.item(),
                    "Identity Loss A": loss_idt_A.item(),
                    "Identity Loss B": loss_idt_B.item(),
                    "GAN Loss A2B": loss_gan_A2B.item(),
                    "GAN Loss B2A": loss_gan_B2A.item()
                })

        # Save model
        if epoch % args.checkpoint_interval == 0:
            torch.save(G_A2B.state_dict(), os.path.join(args.checkpoint_dir, f"G_A2B_baseline_epoch{epoch}.pth"))
            print(f"Saved checkpoint at epoch {epoch}")

        if epoch % args.sample_interval == 0:
            with torch.no_grad():
                fake_B = G_A2B(real_A)
            wandb.log({"Generated Sample": [wandb.Image(fake_B[0].cpu(), caption=f"Epoch {epoch}")]})

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline CycleGAN with wandb tracking")

    # Paths and logging
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project', type=str, default='clip-cyclegan-monet')
    parser.add_argument('--wandb_entity', type=str, default=None)

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--n_res_blocks', type=int, default=9)
    parser.add_argument('--num_workers', type=int, default=4)

    # Loss weights
    parser.add_argument('--lambda_cycle', type=float, default=10.0)
    parser.add_argument('--lambda_identity', type=float, default=5.0)

    # Logging and saving
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--checkpoint_interval', type=int, default=5)

    args = parser.parse_args()

    print("Arguments:", args)
    train(args)