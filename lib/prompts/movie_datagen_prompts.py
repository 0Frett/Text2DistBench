# Data Generation

VIDEO_VALIDATION_TEMPLATE = """
    You searched YouTube for: "{query}"  
    The retrieved video is:  
    Title: {title}  
    Description: {description}  

    Determine if this video matches the intended search query and satisfies the information need.  

    Respond in the following strict JSON format:  
    {{"valid": true}} or {{"valid": false}}
"""


DOC_TEMPLATE = """
    - Movie Title: {title}
    - Release Date: {date}
    - Cast: {casts}
    - Summary: {summary}
    - Synopsis: {synopsis}
"""


ATTRS_TOPIC = ["Actor", "Storyline", "Visual", "Audio", "Other"]

TOPIC_CLF_TEMPLATE = """
    You are analyzing public reactions to a movie by assigning each viewer comment to one or more attributes (multi-label).

    Attributes (use EXACTLY these keys; prefer the single most dominant attribute unless the comment clearly discusses multiple):
    - Actor : Comments about the actors’ performances: delivery, emotion, chemistry, casting.
    - Storyline : Comments about the movie plot, narrative, themes, pacing, character arcs, or dialogue.
    - Visual : Comments about cinematography, animation, lighting, color, production design, costumes, or visual effects.
    - Audio : Comments about the soundtrack, score, songs, sound effects, or audio mix.
    - Other : Use only if none of the above clearly fit (off-topic, spam, unclear).

    --------------------------
    Movie Information:
    {meta_data}

    YouTube Viewer Comments (0-based indexing, e.g., 0, 1, 2, ...):
    {comments}

    Instructions:
    1) MULTI-LABEL is allowed, but if uncertain choose the single most dominant attribute.
    2) Use 0-based indices exactly as shown; do not invent indices.
    3) If a comment does not clearly fit any attribute, include it under "Other".
    4) Return ONLY the JSON object (no markdown, no prose).
    5) The JSON must be a single object whose keys are EXACTLY the attributes below and whose values are lists of integer indices.
    6) Do not add, rename, or remove keys.

    Output JSON (and nothing else):
    {{
        "Actor": [],
        "Storyline": [],
        "Visual": [],
        "Audio": [],
        "Other": []
    }}
"""

STANCE_CLF_TEMPLATE = """
    You classify the stance expressed in YouTube comments toward the movie.

    --------------------------
    Movie Information (for information reference only):
    {meta_data}

    YouTube Viewer Comments (0-based indexing, e.g., 0, 1, 2, ...):
    {comments}

    Labels (choose exactly one per comment):
    - support : praise / approval / positive attitude
    - oppose  : criticism / disapproval / negative attitude

    Rules:
    1) Focus on the overall tone or attitude of the comment toward the movie.
    2) Use movie information only to resolve references (e.g., who/what "he", "it", or "this scene" refers to), not to guess sentiment.
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

