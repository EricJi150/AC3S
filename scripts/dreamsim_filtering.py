import argparse
import os

import torch
from dreamsim import dreamsim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ac3s.dataset import ObjectRenderSyntheticImage

def grayscale(image, device):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device)[None, :, None, None]
    grayscaled_image = (image * weights).sum(dim=1, keepdim=True)
    grayscaled_image = grayscaled_image.repeat(1, 3, 1, 1)
    return grayscaled_image

def dreamsim_metric(model, images, references, device):
    grayscaled_images = grayscale(images, device)
    grayscaled_references = grayscale(references, device)
    return model(grayscaled_images, grayscaled_references)

def main(
    data_directory,
    output_directory,
    cache_dir,
    synsets,
    batch_size,
    device,
):
    model, _ = dreamsim(pretrained=True, device=device, cache_dir=cache_dir)

    if synsets is None or len(synsets) == 0:
        synsets = os.listdir(data_directory)

    os.makedirs(output_directory, exist_ok=True)
    for synset in tqdm(synsets):
        dataset = ObjectRenderSyntheticImage(data_directory, [synset])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        output_file_path = os.path.join(output_directory, f"{synset}.txt")
        with open(output_file_path, "w") as f:
            for render, image, image_paths in tqdm(dataloader, leave=False):
                render = render.to(device)
                image = image.to(device)
                with torch.no_grad():
                    scores = dreamsim_metric(model, image, render, device).cpu()
                for path, score in zip(image_paths, scores):
                    f.write(f"{path},{score.item()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score rendered/synthetic image pairs with DreamSim")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_directory", type=str, default="data/dreamsim_scores", help="Directory to write DreamSim score files to")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/dreamsim", help="Directory to store the pretrained DreamSim weights in")
    parser.add_argument("--synsets", type=str, nargs="+", default=None, help="Synsets to score (default: all found in data_directory)")
    parser.add_argument("--batch_size", type=int, default=256, help="Number of image pairs to score per batch")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run scoring on")
    args = parser.parse_args()

    main(
        args.data_directory,
        args.output_directory,
        args.checkpoint_dir,
        args.synsets,
        args.batch_size,
        args.device,
    )
