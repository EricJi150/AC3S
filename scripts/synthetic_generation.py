import os
import torch
import random
import argparse

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from ControlNetAdapter.model.MLP import MLP
from ControlNetAdapter.utils.dataset import ObjectRender
from ControlNetAdapter.model.model import SDControlNetModel
from ControlNetAdapter.utils.inference import generate, get_conditioning_scale

def parse_args():
    parser = argparse.ArgumentParser(description='Generate DST dataset')
    parser.add_argument('--data_directory', type=str, default='/work/hdd/bdyo/ericji150/Experiments_Renders_2500')
    parser.add_argument('--prompts_directory', type=str, default='/work/hdd/bdyo/ericji150/Experiments_Prompts_2500')
    parser.add_argument('--weights_directory', type=str, default='/u/ericji150/ControlNetAdapter/global_mlp/FeatureMLP.pt')
    parser.add_argument('--synsets', type=str, nargs='+', default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    dataset = ObjectRender(args.data_directory, args.prompts_directory, args.synsets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = SDControlNetModel()
    model.eval()
    mlp = MLP().to("cuda")
    mlp.load_state_dict(torch.load(args.weights_directory)["model_state"])
    mlp.eval()

    seeds = {}
    for image, edge_image, positive_prompt, negative_prompt, paths in tqdm(dataloader):
        image, edge_image = image.to("cuda"), edge_image.to("cuda")

        seed = random.randint(0, 65535)

        with torch.no_grad():
            seed_everything(seed)
            conditioning_scale = get_conditioning_scale(
                model,
                mlp,
                positive_prompt,
                edge_image,
                negative_prompt,
            )
            conditioning_scale = conditioning_scale
            print("Conditioning Scale:", conditioning_scale)
            
            seed_everything(seed)
            output = generate(
                model,
                edge_control_pixel_values=edge_image,
                conditioning_scale=1.0,
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                height = 512,
                width = 512,
                num_inference_steps=20,
                modulated_conditioning_scale = conditioning_scale
            )

            for idx in range(output.shape[0]):
                image = output[idx]
                image = transforms.ToPILImage()(image.cpu())
                dst_path = os.path.join(args.data_directory, paths["synset"][idx], paths["model_id"][idx], "synthetic_image")
                os.makedirs(dst_path, exist_ok=True)
                image.save(os.path.join(dst_path, paths["image_id"][idx]+'.png'))
                print(f"Saved synthetic image to {os.path.join(dst_path, paths['image_id'][idx]+'.png')}")

if __name__ == '__main__':
    main()