"""Filesystem enumeration of inputs, and crash-safe writing of outputs."""
import json
import os
import tempfile
import time
from pathlib import Path


def find_imagenet_images(path, synset):
    synset_path = os.path.join(path, synset)
    if not os.path.exists(synset_path):
        synset_path = os.path.join(path, "train", synset)
    images = []
    for ext in [".JPEG", ".jpeg", ".jpg", ".png"]:
        images.extend(Path(synset_path).glob(f"*{ext}"))
    return sorted([str(p) for p in images])


def find_edge_maps(path, synset):
    synset_path = os.path.join(path, "train", synset)
    if not os.path.exists(synset_path):
        synset_path = os.path.join(path, synset)

    edge_maps = []
    for model_id in sorted(os.listdir(synset_path)):
        model_path = os.path.join(synset_path, model_id)
        if not os.path.isdir(model_path):
            continue
        render_path = os.path.join(model_path, "image_render")
        if not os.path.exists(render_path):
            render_path = model_path
        for f in sorted(os.listdir(render_path)):
            if f.endswith(".png"):
                edge_maps.append((os.path.splitext(f)[0], os.path.join(render_path, f), model_id))
    return edge_maps


def atomic_json_save(data, filepath, max_retries=3, backoff=2.0):
    """Write JSON atomically: write to temp file, then rename into place.

    This prevents 0-byte corruption when the filesystem transport dies
    mid-write (e.g. 'Cannot send after transport endpoint shutdown').
    Retries with exponential backoff on transient OSError.
    """
    out_dir = os.path.dirname(filepath) or "."
    for attempt in range(1, max_retries + 1):
        fd, tmp_path = None, None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=out_dir)
            with os.fdopen(fd, "w") as f:
                fd = None  # os.fdopen takes ownership
                json.dump(data, f, indent=2)
            os.rename(tmp_path, filepath)
            return  # success
        except OSError as e:
            # Clean up temp file
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if attempt < max_retries:
                wait = backoff ** attempt
                print(f"[WARN] Save failed (attempt {attempt}/{max_retries}): {e}. "
                      f"Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] Save failed after {max_retries} attempts: {e}")
                raise
