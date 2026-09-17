import cv2
import numpy as np
import torch

def generate_edge(image, canny_lower=100, canny_upper=200):
    edges = cv2.Canny(image, canny_lower, canny_upper)/255
    edges = edges[None, :, :]
    edges = np.concatenate([edges, edges, edges], axis=0)
    return torch.from_numpy(edges)