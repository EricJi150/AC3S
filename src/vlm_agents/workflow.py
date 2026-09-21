"""LangGraph wiring for the caption pipeline."""
from langgraph.graph import END, StateGraph

from .agents import (
    agent_caption_synthesis,
    agent_negative_prompt,
    agent_parallel_234,
    agent_protected_terms,
    agent_subject_identifier,
)
from .state import CaptionState
from .vlm_agent import set_agent


def build_workflow(vlm=None):
    """
    Build the caption generation workflow.

    Workflow structure:
        Agent1 (subject)
            │
            ▼
        Agent_parallel_234 (genre + pose_camera + appearance_env in parallel)
            │
            ▼
        Agent5 (synthesis) → Agent6 (protected) → Agent7 (negative)

    Nodes reach the VLM through vlm_agent.get_agent() because LangGraph nodes
    receive only `state`. Pass `vlm` here to install it in one step, or call
    vlm_agent.set_agent(...) yourself before invoking the compiled graph.
    """
    if vlm is not None:
        set_agent(vlm)

    workflow = StateGraph(CaptionState)

    workflow.add_node("subject", agent_subject_identifier)
    workflow.add_node("parallel_234", agent_parallel_234)
    workflow.add_node("synthesis", agent_caption_synthesis)
    workflow.add_node("protected", agent_protected_terms)
    workflow.add_node("negative", agent_negative_prompt)

    workflow.set_entry_point("subject")
    workflow.add_edge("subject", "parallel_234")  # Fan-out to parallel agents
    workflow.add_edge("parallel_234", "synthesis")  # Fan-in from parallel agents
    workflow.add_edge("synthesis", "protected")
    workflow.add_edge("protected", "negative")
    workflow.add_edge("negative", END)

    return workflow.compile()
