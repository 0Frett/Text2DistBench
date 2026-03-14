# Data Generation

DOC_TEMPLATE = """
    - Music Video Title: {title}
    - Music Release Date: {date}
    - Music Video Description: {description}
"""


DOC_TRANSLATE_TEMPLATE = """
    Translate the following music video description into {lang}.
    - Do NOT translate the field name "Description".
    - Translate only the value of the description.
    - If the description is already in {lang}, return it unchanged.
    - Return ONLY the JSON object, no extra text.

    Description: {text}

    Output Format:
    Return a single JSON object with this structure:
    {{
        "Description": "<translated value here>"
    }}
"""


ATTRS_TOPIC = ["Song", "Singer", "Lyrics", "Visual", "Other"]

TOPIC_CLF_TEMPLATE = """
    You are analyzing public reactions to a song by assigning each viewer comment to one or more aspects (multi-label).

    Attributes (use EXACTLY these keys; prefer the single most dominant aspect unless the comment clearly discusses multiple):
    - Song : The musical composition itself — melody, harmony, rhythm/groove, sections (verse/chorus/bridge), arrangement.
    - Singer : Anything about the performer — tone/timbre, technique (runs/belts/falsetto), emotion, appearance.
    - Lyrics : The words and meaning — themes, storytelling, message, lines/phrases, rhymes.
    - Visual : Everything visual about the music video — cinematography, lighting/color, concept/storyline, choreography/dance, animation/VFX.
    - Other : Use only if none of the above clearly fit (off-topic, spam, unclear).

    --------------------------
    Music Video Information:
    {meta_data}

    YouTube Viewer Comments (0-based indexing, e.g., 0, 1, 2, ...):
    {comments}

    Instructions:
    1) MULTI-LABEL is allowed, but if uncertain choose the single most dominant aspect.
    2) Use 0-based indices exactly as shown; do not invent indices.
    3) If a comment does not clearly fit any aspect, include it under "Other".
    4) Return ONLY the JSON object (no markdown, no prose).
    5) The JSON must be a single object whose keys are EXACTLY the aspects below and whose values are lists of integer indices.
    6) Do not add, rename, or remove keys.

    Output JSON (and nothing else):
    {{
        "Song": [],
        "Singer": [],
        "Lyrics": [],
        "Visual": [],
        "Other": []
    }}
"""

STANCE_CLF_TEMPLATE = """
    You classify the stance expressed in YouTube comments toward the song or its music video.

    --------------------------
    Music Video Information (for information reference only):
    {meta_data}

    YouTube Viewer Comments (0-based indexing, e.g., 0, 1, 2, ...):
    {comments}

    Labels (choose exactly one per comment):
    - support : praise / approval / positive attitude
    - oppose  : criticism / disapproval / negative attitude

    Rules:
    1) Focus on the overall tone or attitude of the comment toward the music video.
    2) Use the information only to resolve references (e.g., who/what “he”, “it”, or “this part” refers to), not to guess sentiment.
    3) Consider emojis, slang, irony/sarcasm (e.g., quotes, “/s”, exaggeration, laugh reactions).
    4) Use 0-based indices exactly as shown; do not invent or skip indices.
    5) Each index must appear in EXACTLY ONE list (support OR oppose).
    6) Output ONLY valid JSON (no markdown, no prose, no extra keys).

    Output JSON (and nothing else):
    {{
        "support": [],
        "oppose": []
    }}
"""
