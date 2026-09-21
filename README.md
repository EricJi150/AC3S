# AC3S

Official implementation of AC3S.

## AC3S Synthetic Dataset

The full generated synthetic dataset will be released soon.

## Setup

```bash
conda create -n ac3s python=3.10 -y
conda activate ac3s
pip install -e .
pip install -r requirements.txt
```

## Data layout

```
<data_path>/<synset>/<model_id>/image_render/*.png    # CAD edge renders
<imagenet_path>/<synset>/*.JPEG                        # reference photos
```

CAD edge renders are produced separately with
[DST3D](https://github.com/wufeim/DST3D) — follow that repo's instructions to
render your 3D models before running the pipeline below.

## Pipeline

Scripts live in `scripts/`. Run in following order:

```bash
# 1. VLM-generated captions + negative prompts -> prompts/<synset>/prompt_assignments.json
python scripts/generate_text_prompts.py \
    --data_path /path/to/cad_renders --imagenet_path /path/to/imagenet --output_path prompts/

# (only if you sharded step 1 with --num_splits: merge the shards)
python scripts/merge_prompt_assignments.py --output_path prompts/

# 2. Generate data for training visual prompt modulator -> data/modulator_data.pt
python scripts/generate_pseudo_labels.py --data_dir /path/to/cad_renders --prompts_dir prompts/

# 3. Train the visula prompt modulator -> checkpoints/modulator_ckpt.pt
python scripts/train_modulator.py

# 4. Generate synthetic images -> <data_path>/<synset>/<model_id>/synthetic_image/
python scripts/synthetic_generation.py \
    --data_directory /path/to/cad_renders --prompts_directory prompts/

# 5. Score render/synthetic pairs with DreamSim -> data/dreamsim_scores/<synset>.txt
python scripts/dreamsim_filtering.py --data_directory /path/to/cad_renders
```

Each step's output feeds the next step's default input path.

## Citation

If you use this code or dataset, please cite:

```bibtex
@inproceedings{ji2026ac3s,
  title     = {AC3S: Adaptive Conditioning for 3D-Aware Synthetic Data Generation},
  author    = {Ji, Eric and Hu, Qiran and Ma, Wufei and Jain, Sarthak and Li, Yingying and Do, Minh N. and Liu, Yaoyao},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

MIT — see `LICENSE`.
