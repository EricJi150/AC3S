"""Multi-agent VLM prompt generation for ControlNet-conditioned image synthesis.

Seven agents turn a (CAD edge render, reference photograph) pair into one
positive caption and one negative prompt:

    A1 subject      identify the subject from both images
    A2 genre        pick a photography genre               \\
    A3 pose         contact verb + surface + camera angle   > concurrent
    A4 scene        appearance + setting + lighting        /
    A5 synthesis    assemble the caption under a word budget   (pure Python)
    A6 protected    terms that must not appear in negatives
    A7 negative     assemble negatives under a term budget     (pure Python)

Usage:
    from vlm_agents import VLMAgent, build_workflow, initial_state

    app = build_workflow(vlm=VLMAgent(use_quantization=True))
    final = app.invoke(initial_state(image_id, photo_path, cad_path))
    final["generated_caption"], final["generated_negative_prompt"]

Run it over one or more synsets with `python scripts/generate_text_prompts.py`
(see that module for the CLI, and the repo root README for how this fits into
the rest of the pipeline).

Exports resolve lazily (PEP 562), so the light modules stay importable on
machines without the GPU stack: `vlm_agents.CONFIG` or
`from vlm_agents import prompts` need neither torch nor langgraph;
only touching VLMAgent / the workflow / CaptionState pulls in the heavy
dependencies.
"""
import importlib

# Export name -> (submodule, attribute there). Grouped by the dependencies
# that importing the submodule drags in.
_EXPORTS = {
    # no third-party dependencies
    "CONFIG": ("config", "CONFIG"),
    "TIER1_NEGATIVES": ("config", "TIER1_NEGATIVES"),
    "TIER2_NEGATIVES": ("config", "TIER2_NEGATIVES"),
    "TIER3_NEGATIVES": ("config", "TIER3_NEGATIVES"),
    "TIER4_NEGATIVES": ("config", "TIER4_NEGATIVES"),
    "QUALITY_NEGATIVES": ("config", "QUALITY_NEGATIVES"),
    "find_imagenet_images": ("dataset_io", "find_imagenet_images"),
    "find_edge_maps": ("dataset_io", "find_edge_maps"),
    "atomic_json_save": ("dataset_io", "atomic_json_save"),
    # langgraph
    "CaptionState": ("state", "CaptionState"),
    "initial_state": ("state", "initial_state"),
    # torch + transformers
    "VLMAgent": ("vlm_agent", "VLMAgent"),
    "set_agent": ("vlm_agent", "set_agent"),
    "get_agent": ("vlm_agent", "get_agent"),
    # torch + transformers + langgraph + langchain_core
    "build_workflow": ("workflow", "build_workflow"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(f".{module_name}", __name__), attr)
    globals()[name] = value  # cache so the import runs once
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
