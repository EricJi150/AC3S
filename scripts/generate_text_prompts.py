"""CLI entry point: generate captions + negative prompts for one or more synsets.

    python scripts/generate_text_prompts.py \\
        --data_path CAD_ROOT --imagenet_path IMAGENET_ROOT \\
        --output_path OUT --synsets n02110341 \\
        --seed 42 --use_quantization

Writes OUT/<synset>/prompt_assignments.json incrementally (one atomic write
per image, so a killed job resumes instead of restarting).
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from vlm_agents.dataset_io import atomic_json_save, find_edge_maps, find_imagenet_images
from vlm_agents.state import initial_state
from vlm_agents.vlm_agent import VLMAgent
from vlm_agents.workflow import build_workflow


def discover_synsets(data_path):
    """Synset directory names under data_path, matching find_edge_maps's layout."""
    train_path = os.path.join(data_path, "train")
    root = train_path if os.path.isdir(train_path) else data_path
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-agent VLM prompt generation for one or more ImageNet synsets.",
    )
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--imagenet_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--synsets", type=str, nargs="+", default=None,
        help="Synsets to generate prompts for (default: all found in data_path)")
    parser.add_argument("--seed", type=int, default=42)
    # BooleanOptionalAction so --no-use_quantization can actually disable it;
    # the original store_true/default=True made the flag impossible to turn off.
    parser.add_argument("--use_quantization", action=argparse.BooleanOptionalAction,
        default=True,
        help="4-bit NF4 quantization for the VLM (default on; "
             "--no-use_quantization loads bf16 instead)")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--split_id", type=int, default=0,
        help="This invocation's shard index, in [0, num_splits)")
    parser.add_argument("--num_splits", type=int, default=1,
        help="Number of parallel shards to split the synset across (e.g. for a "
             "SLURM array). With the default of 1, this writes directly to "
             "prompt_assignments.json; with >1, each invocation writes "
             "prompt_assignments_<split_id>.json and you merge them afterward "
             "with merge_prompt_assignments.py")
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def plan_work(args, synset):
    """Pair each CAD render with a random ImageNet photo, then take this split's slice.

    The RNG is consumed for every render before slicing, so a given
    (seed, synset) yields the same pairings no matter how many splits run.
    Callers reseed before calling this per synset, so that invariant holds
    regardless of how many other synsets are processed in the same run.
    """
    imagenet_images = find_imagenet_images(args.imagenet_path, synset)
    edge_maps = find_edge_maps(args.data_path, synset)

    all_params = []
    for img_id, img_path, model_id in edge_maps:
        idx = np.random.randint(len(imagenet_images))
        seed = np.random.randint(1000, 10000)
        all_params.append((img_id, img_path, model_id, idx, seed))

    if args.max_images:
        all_params = all_params[:args.max_images]

    total = len(all_params)
    chunk = (total + args.num_splits - 1) // args.num_splits
    start = args.split_id * chunk
    end = min(start + chunk, total)
    return imagenet_images, all_params[start:end]


def load_assignments(assignments_file):
    """Resume from this split's own output, if a previous run left one."""
    if os.path.exists(assignments_file):
        with open(assignments_file, "r") as f:
            return json.load(f)
    return {}


def generate_split(args, synset, app, agent, imagenet_images, params, assignments, assignments_file):
    """Run the workflow over this split, saving after every image."""
    for img_id, img_path, model_id, imagenet_idx, _seed in params:
        filename = f"{model_id}_{img_id}_00.png"

        if not args.regenerate and filename in assignments:
            continue

        final = app.invoke(initial_state(
            image_id=f"{model_id}_{img_id}",
            imagenet_image_path=imagenet_images[imagenet_idx],
            cad_image_path=img_path,
        ))

        assignments[filename] = {
            "synset": synset,
            "model_id": model_id,
            "image_id": img_id,
            "imagenet_file": os.path.basename(imagenet_images[imagenet_idx]),
            "prompt": final["generated_caption"],
            "negative_prompt": final["generated_negative_prompt"],
        }

        atomic_json_save(assignments, assignments_file)

        agent.clear_memory()
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    synsets = args.synsets
    if synsets is None or len(synsets) == 0:
        synsets = discover_synsets(args.data_path)

    agent = VLMAgent(use_quantization=args.use_quantization)
    app = build_workflow(vlm=agent)

    for synset in tqdm(synsets):
        # Reseed per synset so a given (seed, synset) always yields the same
        # render/photo pairings, regardless of what else runs in this batch.
        np.random.seed(args.seed)
        random.seed(args.seed)

        imagenet_images, params = plan_work(args, synset)

        output_dir = os.path.join(args.output_path, synset)
        os.makedirs(output_dir, exist_ok=True)
        if args.num_splits > 1:
            assignments_file = os.path.join(output_dir, f"prompt_assignments_{args.split_id}.json")
        else:
            assignments_file = os.path.join(output_dir, "prompt_assignments.json")
        assignments = load_assignments(assignments_file)

        generate_split(args, synset, app, agent, imagenet_images, params, assignments, assignments_file)


if __name__ == "__main__":
    main()
