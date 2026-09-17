import argparse
import os
import random

from pytorch_lightning import seed_everything
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ac3s.dataset import ObjectRender
from ac3s.inference import get_conditioning_scale
from ac3s.model import Modulator
from stable_diffusion.inference import generate
from stable_diffusion.model import SDControlNetModel

def main(
    data_directory,
    prompts_directory,
    checkpoint_path,
    synsets,
    batch_size,
    height,
    width,
    timesteps,
    padding,
    min_scale,
    max_scale,
    device,
):
    # Initialize the model and dataset
    model = SDControlNetModel(device=device)
    model.eval()
    modulator_checkpoint = torch.load(checkpoint_path, map_location=device)
    modulator = Modulator().to(device)
    modulator.load_state_dict(modulator_checkpoint["model_state"])
    mean = modulator_checkpoint["mean"]
    std = modulator_checkpoint["std"]
    modulator.eval()

    dataset = ObjectRender(data_directory, prompts_directory, synsets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for image, edge_image, positive_prompt, negative_prompt, paths in tqdm(dataloader):
        image, edge_image = image.to(device), edge_image.to(device)

        seed = random.randint(0, 1000)
        with torch.no_grad():
            # Predict conditioning scale per sample
            seed_everything(seed)
            modulator_output = get_conditioning_scale(
                model=model,
                modulator=modulator,
                visual_prompt=edge_image,
                text_prompt=positive_prompt,
                negative_text_prompt=negative_prompt,
                mean=mean,
                std=std,
                height=height,
                width=width,
                num_inference_steps=timesteps,
                padding=padding,
                min_scale=min_scale,
                max_scale=max_scale,
                device=device,
            )

            # Perform generation
            seed_everything(seed)
            output = generate(
                model=model,
                visual_prompt=edge_image,
                text_prompt=positive_prompt,
                negative_text_prompt=negative_prompt,
                conditioning_scale=1.0,
                guidance_scale=5.0,
                height=height,
                width=width,
                timesteps=timesteps,
                collect_feats=False,
                visual_modulator_output=modulator_output,
            )

            # Save images
            output_batch_size = output.shape[0]
            for idx in range(output_batch_size):
                image = output[idx]
                image = transforms.ToPILImage()(image.cpu())
                dst_path = os.path.join(data_directory, paths["synset"][idx], paths["model_id"][idx], "synthetic_image")
                os.makedirs(dst_path, exist_ok=True)
                image.save(os.path.join(dst_path, paths["image_id"][idx]+'.png'))
                print(f"Saved synthetic image to {os.path.join(dst_path, paths['image_id'][idx]+'.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DST dataset")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to renders directory")
    parser.add_argument("--prompts_directory", type=str, required=True, help="Path to prompts directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/modulator_ckpt.pt", help="Path to the trained modulator checkpoint")
    parser.add_argument("--synsets", type=str, nargs="+", default=None, help="Synsets to generate for (default: all found in data_directory)")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of renders to process per batch")
    parser.add_argument("--height", type=int, default=512, help="Generated image height")
    parser.add_argument("--width", type=int, default=512, help="Generated image width")
    parser.add_argument("--timesteps", type=int, default=20, help="Number of denoising steps per generation")
    parser.add_argument("--padding", type=float, default=0.2, help="Offset added to the modulator's predicted conditioning scale")
    parser.add_argument("--min_scale", type=float, default=0.3, help="Minimum conditioning scale allowed after padding/clamping")
    parser.add_argument("--max_scale", type=float, default=1.0, help="Maximum conditioning scale allowed after padding/clamping")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run generation on")
    args = parser.parse_args()

    main(
        args.data_directory,
        args.prompts_directory,
        args.checkpoint_path,
        args.synsets,
        args.batch_size,
        args.height,
        args.width,
        args.timesteps,
        args.padding,
        args.min_scale,
        args.max_scale,
        args.device,
    )