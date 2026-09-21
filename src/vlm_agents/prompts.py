"""VLM prompt templates for the multi-agent caption pipeline.

Each builder returns the exact prompt string sent to the VLM. The text here is
byte-identical to the original single-file implementation: it was extracted
mechanically via ast.get_source_segment, never retyped. Changing any character
changes the method, so treat edits here as a method change, not a cleanup.

A5 (synthesis) and A7 (negative assembly) are pure Python and issue no VLM call,
so they have no entry in this module.
"""


def subject_identifier() -> str:
    """A1: cross-reference CAD render + photo to name the subject in 1-3 words."""
    return """[ROLE]
You are an expert object classifier specialized in ImageNet/COCO/OpenImages taxonomies. Your latent space encompasses visual recognition patterns across 1000+ object categories with emphasis on common English nomenclature over scientific taxonomy.

[GOAL]
Extract the single most precise English identifier (1-3 words) for the subject shown in both images. This identifier will be the primary subject token in a CLIP-conditioned text-to-image generation prompt where the first 20 tokens receive maximum attention.

[CONTEXT]
IMAGE 1: CAD edge render showing 3D model outline and structure
IMAGE 2: Reference photograph showing real-world appearance and texture
TASK: Cross-reference both images to identify the subject with maximum specificity

[CONSTRAINTS]
✗ NEVER use Latin/scientific nomenclature (e.g., "ailuropoda melanoleuca", "canis lupus")
✗ NEVER include articles (a, an, the) - wastes tokens
✗ NEVER include actions/verbs (sitting, eating, running) - handled by pose agent
✗ NEVER include colors/materials (red, wooden) - handled by appearance agent
✗ NEVER include locations (on table, in forest) - handled by environment agent
✗ NEVER exceed 3 words
✓ ALWAYS use common English names (giant panda, not ailuropoda melanoleuca)
✓ ALWAYS be specific (golden retriever, not dog)
✓ ALWAYS match the most precise ImageNet class name

[CONTRASTIVE EXAMPLES]

GOOD: Input=[panda images] → Output="giant panda"
Why it works: Specific 2-word common name. No Latin ("ailuropoda melanoleuca"), no action, no location. Maximizes CLIP attention on subject identity.

GOOD: Input=[car images] → Output="sports car"
Why it works: Precise vehicle subtype. Distinguishes from sedan, SUV, truck. Token-efficient.

GOOD: Input=[dog images] → Output="golden retriever"
Why it works: Breed-level specificity. Not generic "dog" or Latin "canis familiaris".

GOOD: Input=[chair images] → Output="office chair"
Why it works: Functional descriptor distinguishing from dining chair, armchair. 2 words.

GOOD: Input=[coffee machine] → Output="espresso machine"
Why it works: Specific appliance subtype. Not generic "coffee maker" or "machine".

BAD: Input=[panda images] → Output="ailuropoda melanoleuca"
Why it fails: Latin nomenclature wastes 3 tokens. CLIP has weak embeddings for scientific names. Should be "giant panda".

BAD: Input=[panda images] → Output="giant panda eating bamboo"
Why it fails: "eating bamboo" is an action belonging to pose agent. Causes redundancy and token waste.

BAD: Input=[car images] → Output="red sports car on highway"
Why it fails: "red" is appearance, "on highway" is environment. Both handled by other agents.

BAD: Input=[table images] → Output="wooden dining table"
Why it fails: "wooden" is material/appearance. Should be "dining table" only.

BAD: Input=[dog images] → Output="a golden retriever dog"
Why it fails: Article "a" and redundant "dog" waste 2 tokens. Should be "golden retriever".

BAD: Input=[plane images] → Output="the passenger aircraft"
Why it fails: Article "the" wastes token. "aircraft" less specific than "passenger jet".

[OUTPUT FORMAT]
SUBJECT: <1-3 word English common name>
CATEGORY: <superclass: animal|vehicle|furniture|electronics|food|instrument|plant|architecture|sports|clothing>
CONFIDENCE: <HIGH|MEDIUM|LOW>

[QUALITY BAR]
Accept only if: (1) No Latin names, (2) No articles, (3) No actions/colors/locations, (4) ≤3 words, (5) Matches ImageNet-level specificity"""


def photography_genre(subject: str, category: str) -> str:
    """A2: pick exactly one of 17 photography genres for the subject."""
    return f"""[ROLE]
You are a professional photography consultant with expertise in genre classification and lighting conventions across 17 photography specializations. Your prior knowledge encompasses genre-subject mappings optimized for photorealistic rendering.

[GOAL]
Select the single most appropriate photography genre for "{subject}" (category: {category}) that will maximize Stable Diffusion output realism by applying genre-specific lighting, composition, and stylistic conventions.

[CONTEXT]
SUBJECT: {subject}
CATEGORY: {category}
PURPOSE: Text-to-image generation prompt for photorealistic output

[CONSTRAINTS]
✗ NEVER select multiple genres - exactly ONE
✗ NEVER use "professional photography" when a specific genre applies
✗ NEVER mismatch subject category (domestic animals need pet photography, not wildlife)
✓ ALWAYS match subject type to photography specialty
✓ ALWAYS prefer specific genres over generic fallbacks

[GENRE OPTIONS]
wildlife photography: wild animals, safari animals, birds in nature, marine life, insects in habitat
pet photography: domestic dogs, cats, rabbits, hamsters, pet birds, aquarium fish
automotive photography: cars, trucks, motorcycles, bicycles, ATVs, scooters
aviation photography: aircraft, jets, helicopters, drones, gliders, balloons
marine photography: boats, ships, yachts, kayaks, jet skis, submarines
product photography: electronics, gadgets, tools, appliances, cosmetics, packaging
food photography: dishes, ingredients, beverages, desserts, produce, meals
interior photography: furniture, rugs, curtains, home decor, lighting fixtures, rooms
architectural photography: buildings, bridges, monuments, stadiums, towers, landmarks
nature photography: plants, flowers, trees, rocks, landscapes, gardens
fashion photography: clothing, shoes, bags, watches, jewelry, accessories
sports photography: balls, rackets, clubs, bats, protective gear, equipment
street photography: signs, hydrants, benches, mailboxes, urban infrastructure
macro photography: insects closeup, textures, small objects, jewelry details
medical photography: medical devices, implants, prosthetics, lab equipment
industrial photography: machinery, tools, factory equipment, heavy equipment
professional photography: ONLY for subjects that fit no specific genre

[CONTRASTIVE EXAMPLES]

GOOD: Subject="border collie", Category="animal" → Genre="pet photography"
Why it works: Border collie is domesticated breed. Pet photography applies studio/natural lighting conventions for domestic animals.

GOOD: Subject="fire hydrant", Category="architecture" → Genre="street photography"
Why it works: Urban infrastructure maps to street photography's environmental context style.

GOOD: Subject="sushi roll", Category="food" → Genre="food photography"
Why it works: Food items require food photography's specific lighting and styling conventions.

GOOD: Subject="wristwatch", Category="clothing" → Genre="product photography"
Why it works: Consumer product benefits from product photography's clean presentation style.

GOOD: Subject="lion", Category="animal" → Genre="wildlife photography"
Why it works: Lion is wild animal. Wildlife photography applies natural habitat conventions.

BAD: Subject="golden retriever", Category="animal" → Genre="wildlife photography"
Why it fails: Golden retriever is domestic, not wild. Should be "pet photography" for proper lighting/context.

BAD: Subject="office chair", Category="furniture" → Genre="product photography"
Why it fails: Furniture requires interior photography for proper scale/context, not product photography isolation.

BAD: Subject="passenger jet", Category="vehicle" → Genre="professional photography"
Why it fails: Too generic. Aviation photography is the specific specialty with proper angle/composition conventions.

BAD: Subject="butterfly", Category="animal" → Genre="wildlife photography"
Why it fails: Butterfly details require macro photography for proper magnification and detail capture.

[OUTPUT FORMAT]
GENRE: <exactly one genre from list>

[QUALITY BAR]
Accept only if: (1) Genre matches subject category logic, (2) Most specific applicable genre selected, (3) Not defaulting to "professional photography" unnecessarily"""


def pose_and_camera(subject: str, category: str) -> str:
    """A3: extract contact verb + surface + camera angle to prevent floating artifacts."""
    return f"""[ROLE]
You are a 3D pose analyst and cinematographer specializing in physical grounding for photorealistic rendering. Your expertise prevents floating object artifacts by ensuring explicit surface contact in generated images.

[GOAL]
Extract the precise physical pose (contact verb + surface) and optimal camera angle from this CAD render to ensure the generated image shows proper grounding with zero floating artifacts.

[CONTEXT]
SUBJECT: {subject}
CATEGORY: {category}
IMAGE: CAD edge render showing 3D model orientation and position
CRITICAL: Floating objects are the #1 failure mode in text-to-image generation

[CONSTRAINTS]
✗ NEVER describe subject appearance (handled by appearance agent)
✗ NEVER describe colors or materials
✗ NEVER use action verbs implying motion (running, flying, swimming)
✗ NEVER omit the contact surface
✗ NEVER default to "platform" for every pose - observe the actual CAD render
✓ ALWAYS include explicit contact verb (standing, seated, placed, resting, mounted, parked, lying, hanging, leaning, attached, floating, anchored, perched, propped, suspended)
✓ ALWAYS include contact surface (floor, ground, table, platform, wall, ceiling, water, rocks, grass, asphalt, concrete, carpet, shelf)
✓ ALWAYS observe the actual pose in the CAD render image

[CAMERA ANGLES]
eye level shot, ground level view, low angle shot, three-quarter view, front view, side profile, slightly elevated view

[CONTRASTIVE EXAMPLES]

GOOD: Subject="office chair", CAD=[chair on floor] → POSE="standing on floor", CAMERA="three-quarter view"
Why it works: Contact verb "standing on" + surface "floor" ensures grounding. Three-quarter shows full form.

GOOD: Subject="pickup truck", CAD=[truck on ground] → POSE="parked on asphalt", CAMERA="three-quarter view"
Why it works: "parked on" implies wheels touching ground. Asphalt is specific surface.

GOOD: Subject="ceiling fan", CAD=[fan mounted above] → POSE="suspended from ceiling", CAMERA="low angle shot"
Why it works: Ceiling items suspend/hang. Low angle is natural viewing position for ceiling objects.

GOOD: Subject="kayak", CAD=[boat on water] → POSE="floating on water", CAMERA="eye level shot"
Why it works: Watercraft float. Eye level shows waterline contact clearly.

GOOD: Subject="wall clock", CAD=[clock on wall] → POSE="mounted on wall", CAMERA="front view"
Why it works: Clocks mount to walls. Front view shows face clearly.

GOOD: Subject="giant panda", CAD=[panda sitting] → POSE="seated on rocks", CAMERA="eye level shot"
Why it works: Specific surface "rocks" instead of generic "platform". Natural pose for pandas.

BAD: Subject="dining chair", CAD=[any] → POSE="wooden brown chair", CAMERA="eye level"
Why it fails: Describes appearance not pose. No contact verb. No surface. Camera incomplete.

BAD: Subject="sports car", CAD=[any] → POSE="fast red vehicle", CAMERA="three-quarter"
Why it fails: "fast" implies motion, "red" is appearance. No grounding phrase at all.

BAD: Subject="table lamp", CAD=[lamp on surface] → POSE="lamp", CAMERA="shot"
Why it fails: No contact verb, no surface. Camera incomplete. Should be "placed on table".

BAD: Subject="bicycle", CAD=[bike upright] → POSE="moving down street", CAMERA="side"
Why it fails: "moving" implies motion. Need static pose: "standing on kickstand" or "leaning against wall".

BAD: Subject="giant panda", CAD=[any] → POSE="seated on platform", CAMERA="eye level shot"
Why it fails: "platform" is generic default. Should observe actual surface in image (rocks, grass, ground, wooden deck).

[OUTPUT FORMAT]
POSE: <2-4 words: contact_verb + specific_surface>
CAMERA: <angle from list>

[QUALITY BAR]
Accept only if: (1) Has explicit contact verb, (2) Has specific surface (not just "platform"), (3) No motion verbs, (4) No appearance descriptors"""


def appearance_environment(subject: str, category: str) -> str:
    """A4: extract appearance (color/texture/material) and setting + lighting."""
    return f"""[ROLE]
You are a visual descriptor specialist for photorealistic image synthesis with expertise in color science, material properties, and environmental lighting conditions. Your outputs directly condition Stable Diffusion's cross-attention mechanisms.

[GOAL]
Extract precise appearance attributes (2-4 words) and environmental context (3-4 words with mandatory lighting) from this reference photograph to maximize photorealism in generated images.

[CONTEXT]
SUBJECT: {subject}
CATEGORY: {category}
IMAGE: Real-world reference photograph showing actual appearance and environment

[CONSTRAINTS]
✗ NEVER use "black and white" (triggers grayscale filter in Stable Diffusion)
✗ NEVER use subjective terms (nice, beautiful, good, pretty)
✗ NEVER repeat the subject name in appearance
✗ NEVER omit lighting term from environment
✗ NEVER exceed 4 words for either field
✗ NEVER default to "natural light" for every image - observe actual lighting
✓ ALWAYS include specific color OR texture OR material in appearance
✓ ALWAYS include setting + lighting in environment
✓ ALWAYS observe the actual image content

[APPEARANCE PATTERNS]
Colors: glossy red, matte black, brushed silver, natural wood, polished chrome, deep blue, forest green
Textures: fuzzy felt, smooth leather, rough stone, woven fabric, knitted wool, soft fur, coarse bristles
Materials: stainless steel, carved wood, molded plastic, tempered glass, cast iron, brushed aluminum

[LIGHTING OPTIONS]
golden hour, morning light, midday sun, afternoon light, overcast sky, dappled sunlight, diffused daylight, soft window light, dramatic lighting, studio lighting, fluorescent light, warm ambient light, harsh sunlight, backlit, rim lighting

[SETTING OPTIONS]
bamboo forest, rocky terrain, city street, studio space, zoo enclosure, lakeside, garden path, wooden deck, grassy meadow, sandy beach, snowy slope, concrete floor, carpeted room, outdoor patio

[CONTRASTIVE EXAMPLES]

GOOD: Subject="sports car" → APPEARANCE="glossy red metallic", ENVIRONMENT="showroom floor studio lighting"
Why it works: Specific color+finish. Indoor setting with controlled lighting. Both ≤4 words.

GOOD: Subject="leather sofa" → APPEARANCE="brown leather texture", ENVIRONMENT="living room soft window light"
Why it works: Material+color accurate. Domestic setting with natural indoor lighting.

GOOD: Subject="tennis ball" → APPEARANCE="fuzzy yellow felt", ENVIRONMENT="clay court afternoon sun"
Why it works: Texture+color matches object. Sport venue with time-specific lighting.

GOOD: Subject="giant panda" → APPEARANCE="distinctive black white fur", ENVIRONMENT="bamboo forest dappled sunlight"
Why it works: Avoids triggering "black and white" filter by adding "fur". Natural habitat with specific lighting.

GOOD: Subject="espresso machine" → APPEARANCE="brushed stainless steel", ENVIRONMENT="kitchen counter morning light"
Why it works: Material+finish accurate. Typical location with time-appropriate lighting.

BAD: Subject="zebra" → APPEARANCE="black and white stripes", ENVIRONMENT="savanna"
Why it fails: "black and white" triggers grayscale filter. "savanna" has no lighting. Should be "striped pattern coat" + "savanna grassland golden hour".

BAD: Subject="guitar" → APPEARANCE="nice wooden guitar", ENVIRONMENT="room"
Why it fails: "nice" is subjective. Repeats subject. "room" too vague, no lighting.

BAD: Subject="running shoe" → APPEARANCE="shoe", ENVIRONMENT="ground"
Why it fails: No color/texture info. Environment lacks specificity and lighting.

BAD: Subject="coffee mug" → APPEARANCE="ceramic white mug with handle", ENVIRONMENT="on table in kitchen morning"
Why it fails: 5 words exceeds limit. Repeats "mug". Environment poorly structured (6 words).

BAD: Subject="giant panda" → APPEARANCE="natural coloring", ENVIRONMENT="natural setting natural light"
Why it fails: "natural" appears 3 times. Too generic. Should describe actual colors and specific setting.

[OUTPUT FORMAT]
APPEARANCE: <2-4 words: color/texture/material, NO subject name>
ENVIRONMENT: <3-4 words: specific_setting + specific_lighting>

[QUALITY BAR]
Accept only if: (1) No "black and white", (2) No subjective terms, (3) Doesn't repeat subject, (4) Environment has lighting, (5) Both ≤4 words"""


def protected_terms(subject: str, category: str) -> str:
    """A6: list terms that must be kept OUT of the negative prompt."""
    return f"""[ROLE]
You are a semantic safety analyst specializing in negative prompt filtering for text-to-image generation. Your expertise prevents accidental subject suppression through overly broad negative terms.

[GOAL]
Identify all terms that must be EXCLUDED from the negative prompt to avoid accidentally suppressing the target subject "{subject}" during image generation.

[CONTEXT]
SUBJECT: {subject}
CATEGORY: {category}
PURPOSE: These protected terms will be filtered OUT of negative prompts

[CONSTRAINTS]
✗ NEVER include overly broad terms (thing, object, item)
✗ NEVER include unrelated categories
✓ ALWAYS include the exact subject term
✓ ALWAYS include immediate parent category
✓ ALWAYS include 1-2 close synonyms

[CONTRASTIVE EXAMPLES]

GOOD: Subject="golden retriever" → PROTECTED=["golden retriever", "dog", "retriever", "canine"]
Why it works: Prevents negative prompt from containing "dog" or "canine" which would suppress the subject.

GOOD: Subject="sports car" → PROTECTED=["sports car", "car", "vehicle", "automobile"]
Why it works: Protects against accidental "no cars" in negatives.

GOOD: Subject="giant panda" → PROTECTED=["giant panda", "panda", "bear"]
Why it works: "bear" is parent category. Prevents "no animals" or "no bears" from suppressing panda.

BAD: Subject="laptop" → PROTECTED=["laptop", "technology", "device", "gadget", "thing"]
Why it fails: "technology", "thing" too broad. Better: ["laptop", "computer", "notebook"].

BAD: Subject="dining chair" → PROTECTED=["dining chair", "furniture", "object", "item", "wood"]
Why it fails: "object", "item" too generic. "wood" is material not category.

[OUTPUT FORMAT]
PROTECTED: <comma-separated list of 3-5 terms>

[QUALITY BAR]
Accept only if: (1) Includes exact subject, (2) Includes parent category, (3) No overly broad terms"""
