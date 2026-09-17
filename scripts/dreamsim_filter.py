import os
import torch
import argparse
from tqdm import tqdm
from dreamsim import dreamsim
from torch.utils.data import DataLoader
from ControlNetAdapter.utils.dataset import FilteringDataset

def grayscale(image, device="cuda"):
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device)[None, :, None, None]
    grayscaled_image = (image * weights).sum(dim=1)
    grayscaled_image = grayscaled_image.unsqueeze(1).repeat(1,3,1,1)
    return grayscaled_image

def dreamsim_metric(images, references, device="cuda"):
    model, preprocess = dreamsim(pretrained=True, device=device, cache_dir="/u/ericji150/ControlNetAdapter/huggingface pipeline")
    grayscaled_images = grayscale(images)
    grayscaled_references = grayscale(references)
    score = model(grayscaled_images, grayscaled_references)
    return score

def parse_args():
    parser = argparse.ArgumentParser(description='Generate DST dataset')
    parser.add_argument('--data_directory', type=str, default='/work/hdd/bdyo/ericji150/Experiments_Renders_2500')
    parser.add_argument('--output_directory', type=str, default='/work/hdd/bdyo/ericji150/Experiments_Scores_2500')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--synsets', type=str, nargs='+', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)

    for synset in tqdm(args.synsets):
        output_file_path = os.path.join(args.output_directory, f"{synset}.txt")
        with open(output_file_path, 'w') as f:
            dataset = FilteringDataset(args.data_directory, [synset])
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            for render, image, image_path in tqdm(dataloader):
                render = render.to("cuda")
                image = image.to("cuda")
                dreamsim_score = dreamsim_metric(image, render).cpu()
                for idx, image_path in enumerate(image_path):
                    f.write(f"{image_path},{dreamsim_score[idx]}\n")

if __name__ == '__main__':
    main()