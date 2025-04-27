import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import os
import argparse
from PIL import Image
import numpy as np

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
    # Save the image
    image.save(filename)

#Loads models, generates images, saves qualitative and quantitative sets.
def evaluate_models(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Making necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    qualitative_dir = os.path.join(args.output_dir, "qualitative_comparison")
    fid_real_dir = os.path.join(args.output_dir, f"fid_real_{args.target_domain}")
    fid_baseline_dir = os.path.join(args.output_dir, f"fid_fake_{args.target_domain}_baseline")
    fid_clip_dir = os.path.join(args.output_dir, f"fid_fake_{args.target_domain}_clip")

    os.makedirs(qualitative_dir, exist_ok=True)
    os.makedirs(fid_real_dir, exist_ok=True)
    os.makedirs(fid_baseline_dir, exist_ok=True)
    os.makedirs(fid_clip_dir, exist_ok=True)
    print(f"Output directories created/verified in '{args.output_dir}'")

    # Load Models
    print("Loading models...")
    G_baseline = Generator(n_residual_blocks=args.n_res_blocks).to(device)
    G_clip = Generator(n_residual_blocks=args.n_res_blocks).to(device)

    try:
        G_baseline.load_state_dict(torch.load(args.baseline_model_path, map_location=device))
        print(f"Loaded baseline model from: {args.baseline_model_path}")

        # Load CLIP-enhanced checkpoint dictionary
        clip_checkpoint = torch.load(args.clip_model_path, map_location=device)
        # 'G_A2B_state_dict' or 'G_B2A_state_dict' to evaluate
        generator_key = f'G_{args.source_domain}2{args.target_domain}_state_dict'
        if generator_key not in clip_checkpoint:
             if 'G_A2B_state_dict' in clip_checkpoint and args.source_domain == 'A':
                 generator_key = 'G_A2B_state_dict'
             elif 'G_B2A_state_dict' in clip_checkpoint and args.source_domain == 'B':
                 generator_key = 'G_B2A_state_dict'
             else:
                 try:
                     G_clip.load_state_dict(clip_checkpoint)
                     print(f"Loaded CLIP model state dict directly from: {args.clip_model_path}")
                     generator_key = None # Mark as loaded directly
                 except:
                      raise KeyError(f"Could not find a suitable generator state dict key (tried '{generator_key}', 'G_A2B_state_dict', 'G_B2A_state_dict') in {args.clip_model_path}. Check the checkpoint file structure.")

        if generator_key: # Load using the identified key
             G_clip.load_state_dict(clip_checkpoint[generator_key])
             print(f"Loaded CLIP model state dict using key '{generator_key}' from: {args.clip_model_path}")

        G_baseline.eval()
        G_clip.eval()
        print("Models loaded successfully and set to evaluation mode.")

    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
        exit()
    except KeyError as e:
        print(f"Error: Could not find expected key in checkpoint file - {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred loading models: {e}")
        exit()

    # Load Data
    print("Loading dataset...")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    try:
        # Load the SOURCE domain test set for generating images
        source_dataset = Monet2PhotoDataset(
            root_dir=args.dataset_path,
            domain=args.source_domain, # e.g., 'A' for Monet
            transform=transform,
            mode="test" # Use the 'test' split
        )
        source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print(f"Loaded source domain '{args.source_domain}' test set ({len(source_dataset)} images).")

        # Load the TARGET domain test set for saving real images for FID
        target_dataset = Monet2PhotoDataset(
            root_dir=args.dataset_path,
            domain=args.target_domain, # e.g., 'B' for Photo
            transform=transform,
            mode="test" # Use the 'test' split
        )
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        print(f"Loaded target domain '{args.target_domain}' test set ({len(target_dataset)} images).")

    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset path is correct and contains testA/testB folders.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during dataset loading: {e}")
        exit()


    # Generate and Save Images
    print("Generating and saving images...")
    img_counter = 0
    with torch.no_grad():
        # 1. Save real target images for FID
        print(f"Saving real target domain '{args.target_domain}' images for FID...")
        for i, real_target_batch in enumerate(target_loader):
            real_target_batch = real_target_batch.to(device)
            for j in range(real_target_batch.size(0)):
                img_unnormalized = unnormalize(real_target_batch[j])
                save_path = os.path.join(fid_real_dir, f"real_{args.target_domain}_{img_counter:05d}.png")
                save_image_pil(img_unnormalized, save_path)
                img_counter += 1
        print(f"Saved {img_counter} real target images to '{fid_real_dir}'")
        img_counter = 0 # Reset counter for generated images

        # 2. Generate images using source domain inputs
        print(f"Generating fake target domain '{args.target_domain}' images...")
        for i, real_source_batch in enumerate(source_loader):
            real_source = real_source_batch.to(device)

            # Generate fakes
            fake_target_baseline = G_baseline(real_source)
            fake_target_clip = G_clip(real_source)

            # Unnormalize for saving
            real_source_unnorm = unnormalize(real_source)
            fake_target_baseline_unnorm = unnormalize(fake_target_baseline)
            fake_target_clip_unnorm = unnormalize(fake_target_clip)

            # Save images
            for j in range(real_source.size(0)):
                current_img_idx = i * args.batch_size + j

                # Save individual images for FID calculation
                save_path_baseline = os.path.join(fid_baseline_dir, f"fake_{args.target_domain}_baseline_{current_img_idx:05d}.png")
                save_path_clip = os.path.join(fid_clip_dir, f"fake_{args.target_domain}_clip_{current_img_idx:05d}.png")
                save_image_pil(fake_target_baseline_unnorm[j], save_path_baseline)
                save_image_pil(fake_target_clip_unnorm[j], save_path_clip)

                # Save qualitative comparison grid (Input | Baseline | CLIP)
                # Limit number of qualitative images if needed
                if args.num_qualitative < 0 or current_img_idx < args.num_qualitative:
                    comparison_grid = torch.cat((
                        real_source_unnorm[j].unsqueeze(0),
                        fake_target_baseline_unnorm[j].unsqueeze(0),
                        fake_target_clip_unnorm[j].unsqueeze(0)
                    ), dim=0) # Stack along batch dimension temporarily
                    save_path_qualitative = os.path.join(qualitative_dir, f"comparison_{current_img_idx:05d}.png")
                    vutils.save_image(comparison_grid, save_path_qualitative, nrow=3, padding=2) # nrow=3 for side-by-side

            if (i + 1) % 10 == 0:
                print(f"Processed batch {i+1}/{len(source_loader)}")

    print("Image generation complete.")
    print(f"Qualitative comparison images saved in: '{qualitative_dir}'")
    print(f"Images for FID calculation saved in:")
    print(f"  Real Target ({args.target_domain}): '{fid_real_dir}'")
    print(f"  Fake Baseline ({args.target_domain}): '{fid_baseline_dir}'")
    print(f"  Fake CLIP ({args.target_domain}): '{fid_clip_dir}'")

    # FID Calculation Instructions
    print("\n--- FID Calculation Instructions ---")
    print("To calculate FID scores, ensure you have 'pytorch-fid' installed (`pip install pytorch-fid`).")
    print("Then run the following commands in your terminal:")
    print(f"\n# FID for Baseline Model:")
    print(f"python -m pytorch_fid \"{fid_real_dir}\" \"{fid_baseline_dir}\" --device {device}")
    print(f"\n# FID for CLIP-Enhanced Model:")
    print(f"python -m pytorch_fid \"{fid_real_dir}\" \"{fid_clip_dir}\" --device {device}")
    print("\nLower FID scores generally indicate better similarity between generated and real image distributions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and Compare CycleGAN Generators")

    # Paths
    parser.add_argument('--dataset_path', type=str, required=True, help='Root directory of the dataset (containing testA, testB)')
    parser.add_argument('--baseline_model_path', type=str, required=True, help='Path to the baseline generator state_dict file (e.g., G_A2B_baseline.pth)')
    parser.add_argument('--clip_model_path', type=str, required=True, help='Path to the CLIP-enhanced checkpoint file (containing generator state_dict)')
    parser.add_argument('--output_dir', type=str, default='evaluation_output', help='Directory to save evaluation results')

    # Model & Data Config (Should match training)
    parser.add_argument('--source_domain', type=str, default='A', choices=['A', 'B'], help="Source domain ('A' for Monet, 'B' for Photo)")
    parser.add_argument('--target_domain', type=str, default='B', choices=['A', 'B'], help="Target domain ('A' for Monet, 'B' for Photo)")
    parser.add_argument('--img_size', type=int, default=256, help='Image size used during training')
    parser.add_argument('--n_res_blocks', type=int, default=9, help='Number of residual blocks in generators (must match trained models)')

    # Evaluation Config
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num_qualitative', type=int, default=50, help='Number of qualitative comparison images to save (-1 for all)')

    args = parser.parse_args()

    # Basic validation
    if args.source_domain == args.target_domain:
        print("Error: Source and target domains cannot be the same.")
        exit()

    print("Evaluation Arguments:", args)
    evaluate_models(args)
