import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import os
import argparse
from PIL import Image
import sys

from models.generator import Generator
from data.monet2photo_dataset import Monet2PhotoDataset

def unnormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

def save_image_pil(tensor, filename):
    tensor = tensor.cpu().detach()
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(filename)

def evaluate_models(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create output directories
    qualitative_dir = os.path.join(args.output_dir, "qualitative_comparison")
    fid_real_dir = os.path.join(args.output_dir, f"fid_real_{args.target_domain}")
    fid_baseline_dir = os.path.join(args.output_dir, f"fid_fake_{args.target_domain}_baseline")
    fid_clip_dir = os.path.join(args.output_dir, f"fid_fake_{args.target_domain}_clip")
    
    for d in [args.output_dir, qualitative_dir, fid_real_dir, fid_baseline_dir, fid_clip_dir]:
        os.makedirs(d, exist_ok=True)
    print(f"Output directories created/verified at '{args.output_dir}'")

    # Load Models
    G_baseline = Generator(n_residual_blocks=args.n_res_blocks).to(device)
    G_clip = Generator(n_residual_blocks=args.n_res_blocks).to(device)

    try:
        G_baseline.load_state_dict(torch.load(args.baseline_model_path, map_location=device))
        print(f"Loaded baseline model from {args.baseline_model_path}")

        clip_checkpoint = torch.load(args.clip_model_path, map_location=device)
        generator_key = f"G_{args.source_domain}2{args.target_domain}_state_dict"
        if generator_key in clip_checkpoint:
            G_clip.load_state_dict(clip_checkpoint[generator_key])
            print(f"Loaded CLIP model generator using key '{generator_key}'")
        else:
            G_clip.load_state_dict(clip_checkpoint)
            print(f"Loaded CLIP model directly from {args.clip_model_path}")

    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit()

    G_baseline.eval()
    G_clip.eval()

    # Load Datasets
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        source_dataset = Monet2PhotoDataset(
            root_dir=args.dataset_path,
            domain=args.source_domain,
            transform=transform,
            mode="test"
        )
        source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        target_dataset = Monet2PhotoDataset(
            root_dir=args.dataset_path,
            domain=args.target_domain,
            transform=transform,
            mode="test"
        )
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit()

    print(f"Source images loaded: {len(source_dataset)}")
    print(f"Target images loaded: {len(target_dataset)}")

    img_counter = 0

    # Save Real Target Domain Images
    print("Saving real target images...")
    for i, real_target_batch in enumerate(target_loader):
        real_target_batch = real_target_batch.to(device)
        for j in range(real_target_batch.size(0)):
            img_unnormalized = unnormalize(real_target_batch[j])
            save_path = os.path.join(fid_real_dir, f"real_{args.target_domain}_{img_counter:05d}.png")
            save_image_pil(img_unnormalized, save_path)
            img_counter += 1

    img_counter = 0

    # Generate and Save Fakes
    print("Generating and saving fake images...")
    with torch.no_grad():
        for i, real_source_batch in enumerate(source_loader):
            real_source = real_source_batch.to(device)

            fake_baseline = G_baseline(real_source)
            fake_clip = G_clip(real_source)

            real_source_unnorm = unnormalize(real_source)
            fake_baseline_unnorm = unnormalize(fake_baseline)
            fake_clip_unnorm = unnormalize(fake_clip)

            for j in range(real_source.size(0)):
                idx = i * args.batch_size + j

                save_image_pil(fake_baseline_unnorm[j], os.path.join(fid_baseline_dir, f"fake_baseline_{idx:05d}.png"))
                save_image_pil(fake_clip_unnorm[j], os.path.join(fid_clip_dir, f"fake_clip_{idx:05d}.png"))

                # Save qualitative comparison
                if args.num_qualitative < 0 or idx < args.num_qualitative:
                    comparison = torch.cat([
                        real_source_unnorm[j].unsqueeze(0),
                        fake_baseline_unnorm[j].unsqueeze(0),
                        fake_clip_unnorm[j].unsqueeze(0)
                    ], dim=0)
                    save_path = os.path.join(qualitative_dir, f"comparison_{idx:05d}.png")
                    vutils.save_image(comparison, save_path, nrow=3, normalize=True, padding=2)

    print("Image generation complete.")

    print("\n--- FID Calculation Commands ---")
    print(f"FID for Baseline:\npython -m pytorch_fid {fid_real_dir} {fid_baseline_dir} --device {device}")
    print(f"FID for CLIP Model:\npython -m pytorch_fid {fid_real_dir} {fid_clip_dir} --device {device}")
    print("\nLower FID scores are better.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Compare CycleGAN Generators")

    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--baseline_model_path', type=str, required=True)
    parser.add_argument('--clip_model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='evaluation_output')
    parser.add_argument('--source_domain', type=str, default='A', choices=['A', 'B'])
    parser.add_argument('--target_domain', type=str, default='B', choices=['A', 'B'])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--n_res_blocks', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_qualitative', type=int, default=50)

    args = parser.parse_args()

    if args.source_domain == args.target_domain:
        print("Error: source_domain and target_domain must be different!")
        sys.exit()

    print("Evaluation Arguments:", args)
    evaluate_models(args)
