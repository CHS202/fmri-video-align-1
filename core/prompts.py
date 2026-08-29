"""
Prompt construction and response parsing for the two zero-shot tasks:
  - design classification (4-way: MODN / MUJI / SCAN / WABI)
  - experience prediction (regression over N annotation dimensions)

Both prompts ask for strict JSON so parsing doesn't depend on regexing free
text. Adjust ANNOT_SCALE_MIN/MAX below to match your actual rating scale --
placeholder is a 1-7 Likert scale, matching the common convention in scene
perception / affective-rating studies; swap in your real scale if different.
"""
import json
import re

# Shared across all three tasks: what the model is looking at, and the
# guardrails against role-play/speculation. Task-specific instructions
# (classify, rate, describe) are appended on top of this in each
# build_*_prompt() function below.
SYSTEM_PROMPT = (
    "You are a visual evaluation system. Base your assessment only on the "
    "chronologically ordered frames you are given, covering this "
    "80-second architectural interior video. Do not adopt any assumed "
    "profession or role, and do not speculate about rooms, people, "
    "prices, locations, or designer intent that are not shown. Output "
    "only the result in the specified JSON schema, with no reasoning "
    "process or additional text."
)

DESIGN_CLASSES = ["MODN", "MUJI", "SCAN", "WABI"]
DESIGN_CLASS_NAMES = {
    "MODN": "Modern",
    "MUJI": "MUJI (Japanese minimalist)",
    "SCAN": "Scandinavian",
    "WABI": "Wabi-Sabi",
}
CONFIDENCE_SCALE = (1, 5)  # 1 = complete guess, 5 = certain
DEFAULT_SCALE = (1, 7)
DIMENSION_SCALES = {
    "color_comfort": (1, 5),
}

QUESTIONNAIRE = {
    "color_comfort": ("How do you perceive the lighting color in the space?",
                       "Uncomfortable", "Comfortable"),
    "light_association": ("How do you perceive the lighting in the space?",
                           "Cold", "Warm"),
    "complexity": ("This space looks...", "Simple", "Complex"),
    "organization": ("This space looks...", "Disordered", "Organized"),
    "naturalness": ("This space looks...", "Artificial", "Natural"),
    "interest": ("This space looks...", "Boring", "Interesting"),
    "valence": ("This space makes me feel...", "Bad", "Good"),
    "stimulation": ("This space makes me feel...", "Bored", "Interested"),
    "vitality": ("This space makes me feel...", "Lifeless", "Alive"),
    "comfort": ("This space makes me feel...", "Uncomfortable", "Comfortable"),
    "relaxation": ("This space makes me feel...", "Stressed", "Relaxed"),
    "hominess": ("This space makes me feel...", "Alienated", "At home"),
    "uplift": ("This space makes me feel...", "Diminished", "Uplifted"),
    "approachability": ("If I saw this space, I'd...", "Leave", "Enter"),
    "explorability": ("If I saw this space, I'd...", "Ignore it", "Explore it"),
}

def _scale_for(dim):
    return DIMENSION_SCALES.get(dim, DEFAULT_SCALE)

def build_design_prompt():
    # Full style names are given alongside the codes so the model has
    # something concrete to visually ground each label against (the bare
    # codes MODN/MUJI/SCAN/WABI carry no meaning on their own) -- but
    # "proprietary" is kept in the framing so the model treats these names
    # as a hint, not a hard external definition that overrides the actual
    # visual cues in the video.
    class_list = ", ".join(f"{code} ({DESIGN_CLASS_NAMES[code]})" for code in DESIGN_CLASSES)
    conf_lo, conf_hi = CONFIDENCE_SCALE
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "You are watching a first-person walkthrough video of an interior space. "
        f"Classify its design style as exactly one of the following categories: {class_list}. "
        "These are proprietary interior-design style labels; the names in parentheses are "
        "a general guide, not a strict definition -- use the visual cues in the video "
        "(materials, color palette, furniture shape, ornamentation) to decide. "
        "Also report how confident you are in this classification, as an integer from "
        f"{conf_lo} ({conf_lo} = a complete guess) to {conf_hi} ({conf_hi} = certain).\n\n"
        "Respond with strict JSON only, no other text, using ONLY the short code "
        "(not the full name in parentheses) for \"label\":\n"
        f'{{"label": "<one of: {", ".join(DESIGN_CLASSES)}>", '
        f'"confidence": <integer {conf_lo}-{conf_hi}>}}'
    )



def _dimension_line(dim):
    """One numbered line per dimension: the actual question the human raters
    were asked, its low/high anchors, and its numeric range -- rather than
    just the bare dimension name -- so the VLM is answering the same
    perceptual-rating item a human annotator would have seen. Falls back to
    a generic phrasing for any dimension not in QUESTIONNAIRE."""
    lo, hi = _scale_for(dim)
    if dim in QUESTIONNAIRE:
        rating_prompt, low_anchor, high_anchor = QUESTIONNAIRE[dim]
        return (
            f'- "{dim}" -- {rating_prompt} '
            f"Rate from {lo} ({low_anchor}) to {hi} ({high_anchor})."
        )
    return f'- "{dim}" -- rate from {lo} (not at all) to {hi} (extremely).'

def build_experience_prompt(dimension_names):
    # dimensions don't all share one scale (color_comfort is 1-5, most others
    # are 1-7), so each dimension's range -- and now its original question
    # wording and anchors -- is spelled out individually rather than stating
    # one generic scale for the whole list.
    dim_block = "\n".join(_dimension_line(dim) for dim in dimension_names)
    dim_json_fields = ", ".join(f'"{dim}": <rating>' for dim in dimension_names)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "You are watching a first-person walkthrough video of an interior space. "
        "For each item below, answer the question as a typical occupant "
        "would, using the numeric scale and anchor words given for that item.\n\n"
        f"{dim_block}\n\n"
        "Respond with strict JSON only, no other text:\n"
        f"{{{dim_json_fields}}}"
    )

def build_describe_prompt():
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Task: Free description\n"
        "Describe the overall visual impression this space gives you, and "
        "explain which scene features you like or dislike and why. Write "
        "at least 20 words. Describe only what is visible in this clip.\n\n"
        "If occlusion, black frames, or insufficient information prevent a "
        "reasonable assessment, set \"insufficient_visual_evidence\" to "
        "true; still complete the remaining fields based on the visible "
        "evidence.\n\n"
        "Respond with strict JSON only, no other text:\n"
        '{"description": "<at least 20 words>", "insufficient_visual_evidence": <true or false>}'
    )

def _extract_json(text):
    """LMMs frequently wrap JSON in markdown fences or add a leading sentence;
    pull out the first {...} block rather than assuming the whole string parses."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def _match_label_code(label_raw):
    """Matches a raw label string against DESIGN_CLASSES, accepting either
    the bare code ("MUJI") or the code followed by the parenthetical full
    name ("MUJI (Japanese minimalist)") since models sometimes echo the full
    option text from the prompt despite being told to use the short code
    only. Returns None if nothing matches."""
    label_raw = label_raw.strip().upper()
    for code in DESIGN_CLASSES:
        if label_raw == code or label_raw.startswith(code + " ") or label_raw.startswith(code + "("):
            return code
    return None
 
 
def _validate_confidence(raw_val):
    """Returns raw_val as a float if it's within CONFIDENCE_SCALE, else None."""
    conf_lo, conf_hi = CONFIDENCE_SCALE
    try:
        val = float(raw_val)
    except (TypeError, ValueError):
        return None
    return val if conf_lo <= val <= conf_hi else None
 
 
def _regex_extract_design_fields(text):
    """Fallback for when the JSON object never closes (usually: generation
    truncated mid-description). Regexes "label" and "confidence" directly out
    of the raw text without requiring the object as a whole to be
    well-formed -- recovers both fields as long as they were fully emitted
    before truncation, which the label-and-confidence-first schema in
    build_design_prompt() is specifically designed to make likely. Returns
    (label, confidence), either of which may be None if not found/invalid."""
    label = None
    label_match = re.search(r'"label"\s*:\s*"([^"]*)"', text)
    if label_match:
        label = _match_label_code(label_match.group(1))
 
    confidence = None
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    if conf_match:
        confidence = _validate_confidence(conf_match.group(1))
 
    return label, confidence

def parse_design_response(text):
    """Returns (label, confidence, description). label is None if the response
    didn't parse or didn't match one of DESIGN_CLASSES. confidence is None if
    missing, non-numeric, or outside CONFIDENCE_SCALE -- kept separate from
    label validity so a malformed confidence field doesn't throw out an
    otherwise-valid label. description is None if missing/empty or if the
    response was truncated before the JSON object closed (it's the last
    field and the one deliberately sacrificed under truncation -- see module
    docstring); it's not used in any metric, just carried through so you can
    spot-check that the model is actually describing the video content
    rather than pattern-matching straight to a label.
 
    If the full JSON doesn't parse (most often: truncated), falls back to
    regexing label/confidence directly out of the raw text -- this can
    still recover a valid, usable prediction even from an unterminated
    response."""
    obj = _extract_json(text)
    if obj is None:
        label, confidence = _regex_extract_design_fields(text)
        return label, confidence, None
 
    label = None
    if "label" in obj:
        label = _match_label_code(str(obj["label"]))
 
    confidence = _validate_confidence(obj.get("confidence"))
 
    description = obj.get("description")
    description = str(description).strip() if description else None
 
    return label, confidence, description



def normalize_rating(dim, raw_val):
    """Rescale a VLM rating from that dimension's own scale (the range the
    prompt actually asked for -- see DIMENSION_SCALES/DEFAULT_SCALE above) to
    [0, 1], matching the ground-truth annotation scale. Pearson r is
    scale-invariant on its own, but MAE and any raw-value inspection in the
    output CSV need both sides on the same scale, so predictions are
    normalized here rather than left in the VLM's native units."""
    scale_min, scale_max = _scale_for(dim)
    return (raw_val - scale_min) / (scale_max - scale_min)

def _regex_extract_dimension(text, dim):
    """Experience-task counterpart to _regex_extract_design_fields above --
    same failure mode (JSON object truncated mid-generation before it
    closed), but pulls one dimension's numeric value out at a time rather
    than a fixed label/confidence pair, since the field set here is dynamic
    (depends on dimension_names). Matches "<dim>": <number>, tolerating the
    same whitespace variation as the other regex fallbacks. Returns a float,
    or None if the key wasn't found or its value wasn't numeric."""
    match = re.search(rf'"{re.escape(dim)}"\s*:\s*([\d.]+)', text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None
    
def parse_experience_response(text, dimension_names):
    """Returns (result, description). result is a dict of dim -> normalized
    rating in [0, 1] (float('nan') for any dimension that failed to parse),
    or None only if NO dimensions could be recovered at all (neither via a
    full JSON parse nor the regex fallback below). description is the
    model's free-text account of what it saw in the video (None if missing,
    or if the response was truncated before the JSON closed -- it's the last
    field and the one deliberately sacrificed under truncation, see module
    docstring); not used in any metric, carried through purely so you can
    spot-check that ratings are grounded in something the model looked at.
 
    If the full JSON doesn't parse (most often: truncated mid-description),
    each dimension is instead regexed directly out of the raw text -- since
    ratings are listed before description in the schema, this recovers a
    complete rating set from most truncated responses, not just ones that
    happen to close their braces."""
    obj = _extract_json(text)
 
    if obj is not None:
        result = {}
        for dim in dimension_names:
            val = obj.get(dim)
            if val is None:
                # try a case-insensitive fallback match, since models sometimes
                # rephrase keys slightly despite the instruction
                for k, v in obj.items():
                    if k.strip().lower() == dim.strip().lower():
                        val = v
                        break
            try:
                result[dim] = normalize_rating(dim, float(val))
            except (TypeError, ValueError):
                result[dim] = float("nan")
 
        description = obj.get("description")
        description = str(description).strip() if description else None
        return result, description
 
    # Full JSON didn't parse -- regex each dimension out individually.
    result = {}
    n_recovered = 0
    for dim in dimension_names:
        raw_val = _regex_extract_dimension(text, dim)
        if raw_val is None:
            result[dim] = float("nan")
        else:
            result[dim] = normalize_rating(dim, raw_val)
            n_recovered += 1
 
    if n_recovered == 0:
        return None, None
    return result, None

def parse_describe_response(text):
    """Returns (description, insufficient_visual_evidence). description is
    None if the field is missing/empty or the response didn't parse as JSON
    at all (uses the same fenced/leading-text-tolerant _extract_json as the
    other two parsers). insufficient_visual_evidence defaults to False when
    the field is absent, and is coerced to bool when present."""
    obj = _extract_json(text)
    if obj is None:
        return None, None

    description = obj.get("description")
    description = str(description).strip() if description else None

    insufficient = bool(obj.get("insufficient_visual_evidence", False))

    return description, insufficient