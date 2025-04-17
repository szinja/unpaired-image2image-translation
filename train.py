import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_loaders
from tqdm import tqdm
import wandb

def train_fn(disc_A, disc_B, gen_AB, gen_BA, loader, opt_disc, opt_gen, l1_loss, mse_loss, device, config):
    loop = tqdm(loader, leave=True)

    for idx, (real_A, real_B) in enumerate(loop):
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # Train Discriminators A and B
        fake_A = gen_BA(real_B)
        D_A_real = disc_A(real_A.detach())
        D_A_fake = disc_A(fake_A.detach())
        D_A_real_loss = mse_loss(D_A_real, torch.ones_like(D_A_real))
        D_A_fake_loss = mse_loss(D_A_fake, torch.zeros_like(D_A_fake))
        D_A_loss = (D_A_real_loss + D_A_fake_loss) / 2

        fake_B = gen_AB(real_A)
        D_B_real = disc_B(real_B.detach())
        D_B_fake = disc_B(fake_B.detach())
        D_B_real_loss = mse_loss(D_B_real, torch.ones_like(D_B_real))
        D_B_fake_loss = mse_loss(D_B_fake, torch.zeros_like(D_B_fake))
        D_B_loss = (D_B_real_loss + D_B_fake_loss) / 2

        D_loss = (D_A_loss + D_B_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train Generators AB and BA
        D_A_fake = disc_A(fake_A)
        D_B_fake = disc_B(fake_B)
        G_A_fake_loss = mse_loss(D_A_fake, torch.ones_like(D_A_fake))
        G_B_fake_loss = mse_loss(D_B_fake, torch.ones_like(D_B_fake))

        cycle_A = gen_BA(fake_B)
        cycle_B = gen_AB(fake_A)
        cycle_A_loss = l1_loss(real_A, cycle_A) * config.lambda_cycle
        cycle_B_loss = l1_loss(real_B, cycle_B) * config.lambda_cycle

        # Optional: Identity Loss
        if config.use_identity:
            identity_A = gen_BA(real_A)
            identity_B = gen_AB(real_B)
            identity_A_loss = l1_loss(real_A, identity_A) * config.lambda_identity
            identity_B_loss = l1_loss(real_B, identity_B) * config.lambda_identity
            G_loss = (
                G_A_fake_loss
                + G_B_fake_loss
                + cycle_A_loss
                + cycle_B_loss
                + identity_A_loss
                + identity_B_loss
            )
        else:
            G_loss = (
                G_A_fake_loss
                + G_B_fake_loss
                + cycle_A_loss
                + cycle_B_loss
            )

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        # Logging to Wandb
        if idx % 100 == 0:
            wandb.log({
                "D_loss": D_loss.item(),
                "G_loss": G_loss.item(),
                "D_A_loss": D_A_loss.item(),
                "D_B_loss": D_B_loss.item(),
                "G_A_fake_loss": G_A_fake_loss.item(),
                "G_B_fake_loss": G_B_fake_loss.item(),
                "cycle_A_loss": cycle_A_loss.item(),
                "cycle_B_loss": cycle_B_loss.item(),
                "identity_A_loss": identity_A_loss.item() if config.use_identity else 0,
                "identity_B_loss": identity_B_loss.item() if config.use_identity else 0,
            })

        loop.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item())

def main(config):
    # ... (Setup code for models, optimizers, data loaders, etc.)

    for epoch in range(config.num_epochs):
        train_fn(disc_A, disc_B, gen_AB, gen_BA, loader, opt_disc, opt_gen, l1_loss, mse_loss, device, config)
        # ... (Save models, etc.)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_cycle", type=float, default=10)
    parser.add_argument("--lambda_identity", type=float, default=5)
    parser.add_argument("--use_identity", type=bool, default=True)
    # Add any other parameters you want to tune
    config = parser.parse_args()

    wandb.init(project="Monet2Photo-CycleGAN", config=config)
    main(config)
    wandb.finish()