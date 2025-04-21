import torch
import torch.nn as nn
from losses.clip_loss import CLIPLoss

class ModifiedCycleGANLoss(nn.Module):
    def __init__(self, lambda_cycle=10.0, lambda_identity=0.5, lambda_clip=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.clip_loss = CLIPLoss()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_clip = lambda_clip

    def forward(
        self,
        real_A, fake_B, real_B, fake_A,
        cycled_A, cycled_B,
        identity_A=None, identity_B=None,
        target_text_A=None, target_text_B=None,
        disc_real_A=None, disc_fake_A=None,
        disc_real_B=None, disc_fake_B=None
    ):
        # Adversarial loss, for now it's None until we compare CLIP
        if disc_real_A is not None and disc_fake_A is not None:
            D_A_real_loss = self.mse(disc_real_A, torch.ones_like(disc_real_A))
            D_A_fake_loss = self.mse(disc_fake_A, torch.zeros_like(disc_fake_A))
            D_A_loss = (D_A_real_loss + D_A_fake_loss) / 2
        else:
            D_A_loss = 0

        if disc_real_B is not None and disc_fake_B is not None:
            D_B_real_loss = self.mse(disc_real_B, torch.ones_like(disc_real_B))
            D_B_fake_loss = self.mse(disc_fake_B, torch.zeros_like(disc_fake_B))
            D_B_loss = (D_B_real_loss + D_B_fake_loss) / 2
        else:
            D_B_loss = 0

        D_loss = (D_A_loss + D_B_loss) / 2 if (D_A_loss and D_B_loss) else 0

        # Generator adversarial loss
        G_adv_loss_A = self.mse(disc_fake_A, torch.ones_like(disc_fake_A)) if disc_fake_A is not None else 0
        G_adv_loss_B = self.mse(disc_fake_B, torch.ones_like(disc_fake_B)) if disc_fake_B is not None else 0

        # Cycle consistency loss
        cycle_loss_A = self.l1(real_A, cycled_A) * self.lambda_cycle
        cycle_loss_B = self.l1(real_B, cycled_B) * self.lambda_cycle

        # Identity loss
        identity_loss_A = self.l1(real_A, identity_A) * self.lambda_identity if identity_A is not None else 0
        identity_loss_B = self.l1(real_B, identity_B) * self.lambda_identity if identity_B is not None else 0

        # CLIP semantic loss
        clip_loss_A = self.clip_loss.get_loss(fake_B, target_text_B) * self.lambda_clip if target_text_B else 0
        clip_loss_B = self.clip_loss.get_loss(fake_A, target_text_A) * self.lambda_clip if target_text_A else 0

        # Total generator loss
        G_loss = (
            G_adv_loss_A + G_adv_loss_B +
            cycle_loss_A + cycle_loss_B +
            identity_loss_A + identity_loss_B +
            clip_loss_A + clip_loss_B
        )

        return D_loss, G_loss
