import argparse
import os

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

from stable_diffusion.model import SDControlNetModel
from ac3s.dataset import ObjectRender
from stable_diffusion.inference import generate

def main(
    data_directory,
    prompt_directory,
    output_directory,
    output_path,
    num_samples,
    device,
    min_scale,
    max_scale,
    step_size,
    height,
    width,
    timesteps,
    guidance_scale
):
    # Initialize the model and dataset
    model = SDControlNetModel()
    dataset = ObjectRender(data_directory, prompt_directory)

    SD_feats = []
    CN_feats = []
    jnd_scales = []

    # Randomly sample object renders from dataset
    model.eval()
    for sample in tqdm(range(num_samples)):
        image_list = []
        
        data_idx = np.random.randint(len(dataset))
        image, visual_prompt, text_prompt, negative_text_prompt, path = dataset[data_idx]
        image = image.unsqueeze(0).to(device, torch.float32)
        visual_prompt = visual_prompt.unsqueeze(0).to(device, torch.float32)/255
        candidate_scale = torch.arange(min_scale, max_scale + step_size / 2, step_size)
        sample_seed = np.random.randint(0, 1000)

        # Sweep through candidate conditioning
        for cond_scale in candidate_scale:
            torch.manual_seed(sample_seed)
            with torch.no_grad():
                image, feats = generate(
                    model=model,
                    visual_prompt=visual_prompt,
                    text_prompt=text_prompt,
                    negative_text_prompt=negative_text_prompt,
                    conditioning_scale=cond_scale,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    timesteps=timesteps,
                    collect_feats=True,
                    visual_modulator_output=None
                )
            image_list.append(image)

        # JND detection using KMeans clustering
        images_numpy = np.array(torch.cat([image.cpu() for image in image_list], dim=0).flatten(1))
        kmeans = KMeans(n_clusters=2).fit(images_numpy)
        jnd_idx = np.where(kmeans.labels_ == kmeans.labels_[-1])[0][0]

        jnd_scales.append(candidate_scale[jnd_idx].item())
        SD_feats.append(feats[1])
        CN_feats.append(feats[0])
    
    os.makedirs(output_directory, exist_ok=True)
    torch.save(
        {"SD_feats": SD_feats, "CN_feats": CN_feats, "jnd_scale": jnd_scales},
        os.path.join(output_directory, output_path+".pt")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, required=True, help="Path to renders directory")
    parser.add_argument("--prompts_directory", type=str, required=True, help="Path to prompts directory")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory to save output .pt file in")
    parser.add_argument("--output_filename", type=str, required=True, help="Filename to save data as")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to collect")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run generation on")
    parser.add_argument("--min_scale", type=float, default=0.3, help="Minimum conditioning scale to sweep")
    parser.add_argument("--max_scale", type=float, default=1.0, help="Maximum conditioning scale to sweep")
    parser.add_argument("--step_size", type=float, default=0.1, help="Step size between conditioning scales")
    parser.add_argument("--height", type=int, default=512, help="Generated image height")
    parser.add_argument("--width", type=int, default=512, help="Generated image width")
    parser.add_argument("--timesteps", type=int, default=20, help="Number of denoising steps per generation")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale for generation")
    args = parser.parse_args()

    main(
        args.data_directory,
        args.prompts_directory,
        args.output_directory,
        args.output_filename,
        args.num_samples,
        args.device,
        args.min_scale,
        args.max_scale,
        args.step_size,
        args.height,
        args.width,
        args.timesteps,
        args.guidance_scale
    )