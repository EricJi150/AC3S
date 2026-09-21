"""Merge one synset's `prompt_assignments_<split_id>.json` shards, written by
generate_text_prompts.py's --split_id/--num_splits sharding, into the single
prompt_assignments.json that ObjectRender expects to read.

    python scripts/merge_prompt_assignments.py \\
        --output_path out/ --synset n02110341
"""
import argparse
import glob
import json
import os

from vlm_agents.dataset_io import atomic_json_save


def merge_synset(output_path, synset):
    synset_dir = os.path.join(output_path, synset)
    shard_paths = sorted(glob.glob(os.path.join(synset_dir, "prompt_assignments_*.json")))
    if not shard_paths:
        print(f"[{synset}] no prompt_assignments_*.json shards found in {synset_dir}, skipping")
        return

    merged = {}
    for shard_path in shard_paths:
        with open(shard_path, "r") as f:
            shard = json.load(f)
        for filename, record in shard.items():
            if filename in merged and merged[filename] != record:
                print(f"[{synset}] WARNING: {filename} differs between shards; "
                      f"keeping the one from {os.path.basename(shard_path)}")
            merged[filename] = record

    merged_path = os.path.join(synset_dir, "prompt_assignments.json")
    atomic_json_save(merged, merged_path)
    print(f"[{synset}] merged {len(shard_paths)} shards ({len(merged)} images) -> {merged_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge generate_text_prompts.py's per-split prompt_assignments_<split>.json "
                     "shards into the single prompt_assignments.json ObjectRender reads."
    )
    parser.add_argument("--output_path", type=str, required=True, help="Directory containing <synset>/ subdirectories")
    parser.add_argument("--synsets", type=str, nargs="+", default=None, help="Synsets to merge (default: all found in output_path)")
    args = parser.parse_args()

    synsets = args.synsets
    if synsets is None or len(synsets) == 0:
        synsets = sorted(os.listdir(args.output_path))

    for synset in synsets:
        merge_synset(args.output_path, synset)


if __name__ == "__main__":
    main()
