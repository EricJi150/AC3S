"""The state object threaded through every agent in the LangGraph workflow.

Each agent returns a *copy* with its own fields filled in, so the graph can run
A2/A3/A4 concurrently without them clobbering each other's writes.
"""
from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class CaptionState(TypedDict):
    image_id: str
    imagenet_image_path: str
    cad_image_path: str
    subject_label: str
    subject_confidence: float
    subject_category: str
    photography_genre: str
    pose_description: str
    contact_verb: str
    surface: str
    camera_angle: str
    environment: str
    appearance: str
    protected_terms: List[str]
    generated_caption: str
    generated_negative_prompt: str
    messages: Annotated[list, add_messages]


def initial_state(image_id: str, imagenet_image_path: str, cad_image_path: str) -> CaptionState:
    """A zero-valued CaptionState with only the three input paths populated."""
    return {
        "image_id": image_id,
        "imagenet_image_path": imagenet_image_path,
        "cad_image_path": cad_image_path,
        "subject_label": "",
        "subject_confidence": 0.0,
        "subject_category": "",
        "photography_genre": "",
        "pose_description": "",
        "contact_verb": "",
        "surface": "",
        "camera_angle": "",
        "environment": "",
        "appearance": "",
        "protected_terms": [],
        "generated_caption": "",
        "generated_negative_prompt": "",
        "messages": [],
    }
