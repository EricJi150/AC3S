import os
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from ac3s.utils import generate_edge

class ModulatorDataset(Dataset):
    def __init__(self, data_file):
        self.magnitudes = []
        self.conditioning_scales = []

        data = torch.load(data_file)
        for CN_feats, SD_feats, jnd_scale in zip(data["CN_feats"], data["SD_feats"], data["jnd_scale"]):
            self.magnitudes.append(torch.cat([CN_feats, SD_feats]))
            self.conditioning_scales.append(jnd_scale)

        self.magnitudes = torch.stack(self.magnitudes)
        self.conditioning_scales = torch.tensor(self.conditioning_scales).unsqueeze(1)

    def __len__(self):
        return len(self.magnitudes)

    def __getitem__(self, idx):
        return self.magnitudes[idx], self.conditioning_scales[idx]

class ObjectRender(Dataset):
    def __init__(self, data_directory, prompts_directory, synsets=None):
        if synsets is None or len(synsets) == 0:
            synsets = os.listdir(os.path.join(data_directory))

        self.paths = []
        self.images = []
        self.positive_prompts = []
        self.negative_prompts = []        
        
        for synset in synsets:
            with open(os.path.join(prompts_directory, synset, "prompt_assignments.json"), "r") as f:
                prompts = json.load(f)
                
            synset_directory = os.path.join(data_directory, synset)
            object_models = [model for model in os.listdir(synset_directory) if os.path.isdir(os.path.join(synset_directory, model))]
            for object_model in object_models:
                render_directory = os.path.join(synset_directory, object_model, "image_render")
                renders = os.listdir(render_directory)
                for render in renders:
                    self.images.append(os.path.join(render_directory, render))
                    self.paths.append({"synset": synset, "model_id": object_model, "image_id": render.split(".")[0]})
                    for item in prompts.values():
                        if item["synset"] == synset and item["model_id"] == object_model and item["image_id"] == render.split(".")[0]:
                            self.positive_prompts.append(item["prompt"])
                            self.negative_prompts.append(item["negative_prompt"])
                            break
                    
    def __len__(self):
        assert len(self.images) == len(self.positive_prompts) == len(self.negative_prompts) == len(self.paths)
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        edge_image = generate_edge(image).float()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensor()(image).float()
        positive_prompt = self.positive_prompts[idx]
        negative_prompt = self.negative_prompts[idx]
        path = self.paths[idx]
        return image, edge_image, positive_prompt, negative_prompt, path

class FilteringDataset(Dataset):
    def __init__(self, data_directory, synsets=None):
        if synsets is None or len(synsets) == 0:
            synsets = os.listdir(os.path.join(data_directory))

        self.renders = []
        self.images = []
        
        for synset in synsets:                
            synset_directory = os.path.join(data_directory, synset)
            object_models = [model for model in os.listdir(synset_directory) if os.path.isdir(os.path.join(synset_directory, model))]
            for object_model in object_models:
                render_directory = os.path.join(synset_directory, object_model, "image_render")
                image_directory = os.path.join(synset_directory, object_model, "synthetic_image")
                renders = sorted(os.listdir(render_directory))
                images = sorted(os.listdir(image_directory))
                for render in renders:
                    self.renders.append(os.path.join(render_directory, render))
                for image in images:
                    self.images.append(os.path.join(image_directory, image))
                    
    def __len__(self):
        assert len(self.renders) == len(self.images)
        return len(self.renders)

    def __getitem__(self, idx):
        render_path = self.renders[idx]
        render = cv2.imread(render_path, cv2.IMREAD_COLOR)
        render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
        render = ToTensor()(render).float()
        
        image_path = self.images[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensor()(image).float()

        return render, image, image_path
