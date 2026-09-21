"""Tunable limits and the tiered negative-prompt vocabulary.

The negative terms are split into tiers because the assembly step (A7) has a
fixed term budget: tier 1 is always kept, later tiers are admitted only while
budget remains. Tier order therefore encodes priority, not category.
"""

CONFIG = {
    "target_prompt_words": 14,
    "max_prompt_words": 18,
    "quality_suffix": "RAW photo, DSLR, accurate coloring",
    "max_negative_terms": 40,
}

# Tier 1: floating artifacts, the #1 failure mode. Never dropped for budget.
TIER1_NEGATIVES = [
    "floating in air", "levitating", "suspended", "hovering", "no ground contact", "mid-air",
]

TIER2_NEGATIVES = [
    "aerial view", "bird's eye view", "drone shot", "top-down view",
    "3D model", "computer graphics", "plastic texture", "CG render", "video game",
]

TIER3_NEGATIVES = [
    "warm filter", "orange tint", "sepia", "yellow cast", "brown tint",
    "monochrome", "grayscale", "black and white filter", "colorless", "desaturated",
]

TIER4_NEGATIVES = [
    "plain background", "empty background", "solid color background", "white background",
    "studio backdrop", "neon colors", "surreal background", "oversaturated",
    "cartoon", "illustration", "painting", "watercolor", "digital art", "sketch",
]

QUALITY_NEGATIVES = ["blurry", "low quality", "distorted", "deformed", "bad anatomy"]
