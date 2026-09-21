"""The seven agents, A1-A7.

Each is a LangGraph node: it takes a CaptionState, returns a modified copy.
A1/A2/A3/A4/A6 call the VLM and parse its reply; A5 and A7 are pure Python.

Every parser is written to survive a malformed reply by falling back to a
sane default rather than raising, because one unparseable line should not
kill a 50-image batch mid-run.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import AIMessage

from . import prompts
from .config import (
    CONFIG,
    QUALITY_NEGATIVES,
    TIER1_NEGATIVES,
    TIER2_NEGATIVES,
    TIER3_NEGATIVES,
    TIER4_NEGATIVES,
)
from .state import CaptionState
from .vlm_agent import get_agent

VALID_CATEGORIES = [
    "animal", "vehicle", "furniture", "electronics", "food",
    "instrument", "plant", "architecture", "sports", "clothing",
]

VALID_GENRES = [
    "wildlife photography", "pet photography", "automotive photography", "aviation photography",
    "marine photography", "product photography", "food photography", "interior photography",
    "architectural photography", "nature photography", "fashion photography", "sports photography",
    "street photography", "macro photography", "medical photography", "industrial photography",
    "professional photography",
]

# Terms too broad to belong in a negative prompt: dropping them protects the
# subject from being suppressed by its own superclass.
GENERIC_PROTECTED = {
    "animal", "vehicle", "furniture", "object", "food", "plant", "device", "equipment",
}

# Appearance phrasings that trigger a grayscale filter in Stable Diffusion.
GRAYSCALE_TRIGGERS = ["black and white", "b&w", "b/w", "monochrome"]

# Contact verbs are 2 words ("standing on"); these are the linking words we split on.
CONTACT_PREPOSITIONS = ("on", "from", "against")


def _field(line: str) -> str:
    """Value of a `LABEL: value` reply line, stripped of quotes and punctuation."""
    return line.split(":", 1)[1].strip().strip('"\'.,<>')


def _next_state(state: CaptionState, note: str, **fields) -> CaptionState:
    """A copy of state with `fields` applied and `note` appended to the transcript."""
    new_state = state.copy()
    new_state.update(fields)
    new_state["messages"] = state["messages"] + [AIMessage(content=note)]
    return new_state


def agent_subject_identifier(state: CaptionState) -> CaptionState:
    """Agent 1: Subject Identification with Type System Prompt Framework"""
    result = get_agent().analyze_two_images(
        state["cad_image_path"], state["imagenet_image_path"],
        prompts.subject_identifier(),
    )

    subject = "unknown"
    category = "object"
    confidence = 0.5

    for line in result.split('\n'):
        line_lower = line.strip().lower()
        if line_lower.startswith("subject:"):
            subject = _field(line)
            for prefix in ["a ", "an ", "the "]:
                if subject.lower().startswith(prefix):
                    subject = subject[len(prefix):]
            subject = subject.lower()
        elif line_lower.startswith("category:"):
            cat = line.split(":", 1)[1].strip().lower().strip('<>')
            if cat in VALID_CATEGORIES:
                category = cat
        elif line_lower.startswith("confidence:"):
            conf = line.split(":", 1)[1].strip().lower()
            confidence = 0.95 if "high" in conf else 0.75 if "medium" in conf else 0.55

    return _next_state(
        state, f"Subject: {subject} [{category}]",
        subject_label=subject, subject_category=category, subject_confidence=confidence,
    )


def agent_photography_genre(state: CaptionState) -> CaptionState:
    """Agent 2: Photography Genre Selection with Type System Prompt Framework"""
    subject = state.get("subject_label", "subject")
    category = state.get("subject_category", "object")

    result = get_agent().analyze_text(
        prompts.photography_genre(subject=subject, category=category),
    )

    genre = "professional photography"
    for line in result.split('\n'):
        if line.lower().strip().startswith("genre:"):
            g = line.split(":", 1)[1].strip().lower().strip('<>')
            if g in VALID_GENRES:
                genre = g
            else:
                # Reply wasn't verbatim: match on the leading word ("wildlife ...").
                for v in VALID_GENRES:
                    if v.split()[0] in g:
                        genre = v
                        break

    return _next_state(state, f"Genre: {genre}", photography_genre=genre)


def agent_pose_and_camera(state: CaptionState) -> CaptionState:
    """Agent 3: Pose and Camera Angle Extraction with Type System Prompt Framework"""
    subject = state.get("subject_label", "subject")
    category = state.get("subject_category", "object")

    result = get_agent().analyze_image(
        state["cad_image_path"], prompts.pose_and_camera(subject=subject, category=category),
    )

    pose = "resting on surface"
    camera = "eye level shot"
    contact_verb = "resting on"
    surface = "surface"

    for line in result.split('\n'):
        line_lower = line.strip().lower()
        if line_lower.startswith("pose:"):
            pose = _field(line)
            parts = pose.split()
            if len(parts) >= 3 and parts[1] in CONTACT_PREPOSITIONS:
                contact_verb = f"{parts[0]} {parts[1]}"
                surface = " ".join(parts[2:])
        elif line_lower.startswith("camera:"):
            camera = _field(line)

    return _next_state(
        state, f"Pose: {pose}, Camera: {camera}",
        pose_description=pose, contact_verb=contact_verb,
        surface=surface, camera_angle=camera,
    )


def agent_appearance_environment(state: CaptionState) -> CaptionState:
    """Agent 4: Appearance and Environment Extraction with Type System Prompt Framework"""
    subject = state.get("subject_label", "subject")
    category = state.get("subject_category", "object")

    result = get_agent().analyze_image(
        state["imagenet_image_path"], prompts.appearance_environment(subject=subject, category=category),
    )

    appearance = "natural coloring"
    environment = "natural setting soft light"

    for line in result.split('\n'):
        line_lower = line.strip().lower()
        if line_lower.startswith("appearance:"):
            appearance = _field(line)
            for trigger in GRAYSCALE_TRIGGERS:
                if trigger in appearance.lower():
                    appearance = appearance.lower().replace(trigger, "distinctive markings")
        elif line_lower.startswith("environment:"):
            environment = _field(line)

    return _next_state(
        state, f"Appearance: {appearance}, Env: {environment}",
        appearance=appearance, environment=environment,
    )


def agent_caption_synthesis(state: CaptionState) -> CaptionState:
    """Agent 5: Caption Synthesis with CLIP Token Optimization.

    Assembles the caption, then sheds the least load-bearing clause twice if it
    busts the word budget: first the camera angle, then the genre.
    """
    subject = state.get("subject_label", "unknown")
    pose = state.get("pose_description", "resting on surface")
    camera = state.get("camera_angle", "eye level shot")
    environment = state.get("environment", "natural setting")
    genre = state.get("photography_genre", "professional photography")

    quality = f"{genre}, {CONFIG['quality_suffix']}"
    caption = f"{subject} {pose}, {camera}, {environment}, {quality}"

    if len(caption.split()) > CONFIG["max_prompt_words"]:
        caption = f"{subject} {pose}, {environment}, {quality}"

    if len(caption.split()) > CONFIG["max_prompt_words"]:
        caption = f"{subject} {pose}, {environment}, {CONFIG['quality_suffix']}"

    return _next_state(
        state, f"Caption: {len(caption.split())}w", generated_caption=caption,
    )


def agent_protected_terms(state: CaptionState) -> CaptionState:
    """Agent 6: Protected Terms Extraction with Type System Prompt Framework"""
    subject = state.get("subject_label", "")
    category = state.get("subject_category", "object")

    result = get_agent().analyze_text(
        prompts.protected_terms(subject=subject, category=category),
    )

    protected_terms = [subject]
    for line in result.split('\n'):
        if line.lower().strip().startswith("protected:"):
            terms_str = line.split(":", 1)[1].strip().strip('[]')
            terms = [t.strip().strip('"\'') for t in terms_str.split(',')]
            protected_terms = [t for t in terms if t and len(t) > 1]
            break

    if not protected_terms:
        protected_terms = [subject, category]

    return _next_state(
        state, f"Protected: {len(protected_terms)} terms", protected_terms=protected_terms,
    )


def agent_negative_prompt(state: CaptionState) -> CaptionState:
    """Agent 7: Negative Prompt Assembly with Protected Term Filtering.

    Tier 1 (floating artifacts) and the quality terms are unconditional; tiers
    2-4 fill whatever budget is left, in priority order.
    """
    protected = state.get("protected_terms", [])
    subject = state.get("subject_label", "").lower()

    protected_lower = [p.lower() for p in protected] if isinstance(protected, list) else protected.lower().split(',')
    subj_words = set(subject.split())

    def keep(term: str) -> bool:
        """A negative term is safe only if it cannot suppress the subject."""
        t_lower = term.lower()
        if t_lower in protected_lower:
            return False
        if any(w in t_lower for w in subj_words):
            return False
        return t_lower not in GENERIC_PROTECTED

    def filter_terms(terms):
        return [t for t in terms if keep(t)]

    final = filter_terms(TIER1_NEGATIVES) + filter_terms(QUALITY_NEGATIVES)
    budget = CONFIG["max_negative_terms"] - len(final)

    for tier in [TIER2_NEGATIVES, TIER3_NEGATIVES, TIER4_NEGATIVES]:
        if budget > 0:
            add = filter_terms(tier)[:budget]
            final.extend(add)
            budget -= len(add)

    seen = set()
    unique = []
    for n in final:
        if n.lower() not in seen:
            seen.add(n.lower())
            unique.append(n)

    return _next_state(
        state, f"Neg: {len(unique)} terms", generated_negative_prompt=", ".join(unique),
    )


def agent_parallel_234(state: CaptionState) -> CaptionState:
    """
    Parallel execution wrapper for Agents 2, 3, and 4.
    Uses ThreadPoolExecutor to run genre, pose_camera, and appearance_env concurrently.
    While GPU inference is sequential due to shared model, this provides concurrent scheduling
    and reduces overhead from graph traversal.
    """
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(agent_photography_genre, state): "genre",
            executor.submit(agent_pose_and_camera, state): "pose",
            executor.submit(agent_appearance_environment, state): "appearance",
        }
        results = {}
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    new_state = state.copy()

    if "genre" in results:
        new_state["photography_genre"] = results["genre"]["photography_genre"]

    if "pose" in results:
        new_state["pose_description"] = results["pose"]["pose_description"]
        new_state["contact_verb"] = results["pose"]["contact_verb"]
        new_state["surface"] = results["pose"]["surface"]
        new_state["camera_angle"] = results["pose"]["camera_angle"]

    if "appearance" in results:
        new_state["appearance"] = results["appearance"]["appearance"]
        new_state["environment"] = results["appearance"]["environment"]

    # Each branch started from the same state, so its new messages are whatever
    # it appended past the shared prefix.
    prior = len(state.get("messages", []))
    all_messages = []
    for key in ["genre", "pose", "appearance"]:
        if key in results:
            all_messages.extend(results[key].get("messages", [])[prior:])

    new_state["messages"] = state["messages"] + all_messages
    return new_state
