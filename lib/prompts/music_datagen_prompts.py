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



# QA TEMPLATES
TARGETS = ["Song", "Singer", "Lyrics", "Visual"]

# Distribution Estimation templates
EST_SYS_TEMPLATE = """
    You are an expert in analyzing public opinions about a specific music video based on the given YouTube viewer comments.

    Each comment can be interpreted along two dimensions:
    - **Stance** — the expressed attitude toward the music video or its aspects.
    - **Target** — the specific aspect or attribute of the music video being discussed.

    Stances include:
    - "support": positive or approving opinions.
    - "oppose": negative or critical opinions.

    Targets include:
    - Song: musical composition, melody, harmony, rhythm, structure (verse/chorus/bridge), or arrangement.
    - Singer: vocal tone, technique (runs/belts/falsetto), emotion, or appearance.
    - Lyrics: words and meaning, themes, storytelling, message, or rhyme.
    - Visual: cinematography, lighting, color, concept, choreography, or animation/VFX.
    
    You will be asked to estimate quantitative distributions of stance and target using the provided information.

    Music Video Information:
    {meta_data}

    YouTube Viewer Comments:
    {comments}
"""

EST_Hint1 = """
    Important:
    - Base your estimates only on the provided comments.
    - Do not assume labels that do not appear.
    - If a label is absent in the comments, its percentage should reflect that.
"""

EST_Hint2 = """
    Suggested process:
    1. Identify the appropriate label for each comment based on the task definition.
    2. Estimate how many comments belong to each label.
    3. Convert these counts into integer percentages that sum to 100.
"""

# ---------- P(S): Marginal stance distribution ----------
EST_S_TEMPLATE = """
    Question:
    Estimate how YouTube viewers’ comments distribute across different stances toward the music video "{title}".
    Ensure the percentages sum to 100 and use integers only.

    {hints}

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "support": "<int>%",
            "oppose": "<int>%"
        }}
    }}

    Answer:
"""


# ---------- P(T): Marginal target distribution ----------
EST_T_TEMPLATE = """
    Question:
    Estimate how YouTube viewers’ comments distribute across the different targets discussed in the music video "{title}".
    Ensure the percentages sum to 100 and use integers only.

    {hints}

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "Song": "<int>%",
            "Singer": "<int>%",
            "Lyrics": "<int>%",
            "Visual": "<int>%"
        }}
    }}

    Answer:
"""


# ---------- P(S,T): Joint stance–target distribution ----------
EST_T_S_TEMPLATE = """
    Question:
    Estimate the joint distribution of stances and targets among YouTube comments for the music video "{title}".
    Ensure the percentages sum to 100 and use integers only.

    {hints}

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "(Song,support)": "<int>%",
            "(Song,oppose)": "<int>%",
            "(Singer,support)": "<int>%",
            "(Singer,oppose)": "<int>%",
            "(Lyrics,support)": "<int>%",
            "(Lyrics,oppose)": "<int>%",
            "(Visual,support)": "<int>%",
            "(Visual,oppose)": "<int>%"
        }}
    }}

    Answer:
"""


# ---------- P(S∣T): Stance given target ----------
EST_S_cond_T_TEMPLATE = """
    Question:
    Among comments discussing the {target} in the music video "{title}", estimate how viewers’ stances distribute.
    Ensure the percentages sum to 100 and use integers only.

    {hints}

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "support": "<int>%",
            "oppose": "<int>%"
        }}
    }}

    Answer:
"""


# ---------- P(T∣S): Target given stance ----------
EST_T_cond_S_TEMPLATE = """
    Question:
    Among {stance_label} comments toward the music video "{title}", estimate how these comments distribute across different targets.
    Ensure the percentages sum to 100 and use integers only.

    {hints}

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "Song": "<int>%",
            "Singer": "<int>%",
            "Lyrics": "<int>%",
            "Visual": "<int>%"
        }}
    }}

    Answer:
"""


#######

EST_PRIOR_SYS_TEMPLATE = """
    You are an expert in predicting YouTube viewer opinions about a specific music video based on its information.

    Each viewer’s opinion can be interpreted along two dimensions:
    - **Stance** — the expressed attitude toward the music video or its aspects.
    - **Target** — the specific aspect or attribute of the music video being discussed.

    Stances include:
    - "support": positive or approving opinions.
    - "oppose": negative or critical opinions.

    Targets include:
    - Song: musical composition, melody, harmony, rhythm, structure (verse/chorus/bridge), or arrangement.
    - Singer: vocal tone, technique (runs/belts/falsetto), emotion, or appearance.
    - Lyrics: words and meaning, themes, storytelling, message, or rhyme.
    - Visual: cinematography, lighting, color, concept, choreography, or animation/VFX.
    
    You will be asked to predict quantitative distributions of stance and target using the music video information.

    Music Video Information:
    {meta_data}
"""



# ---------- P(S): Marginal stance distribution ----------
EST_PRIOR_S_TEMPLATE = """
    Question:
    Predict how YouTube viewers would distribute across different stances toward the music video "{title}".
    Ensure the percentages sum to 100 and use integers only.

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "support": "<int>%",
            "oppose": "<int>%"
        }}
    }}

    Answer:
"""


# ---------- P(T): Marginal target distribution ----------
EST_PRIOR_T_TEMPLATE = """
    Question:
    Predict how YouTube viewers would distribute across the different targets discussed in the music video "{title}".
    Ensure the percentages sum to 100 and use integers only.

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "Song": "<int>%",
            "Singer": "<int>%",
            "Lyrics": "<int>%",
            "Visual": "<int>%"
        }}
    }}

    Answer:
"""


# ---------- P(S,T): Joint stance–target distribution ----------
EST_PRIOR_T_S_TEMPLATE = """
    Question:
    Predict the joint distribution of stances and targets among YouTube viewers for the music video "{title}".
    Ensure the percentages sum to 100 and use integers only.

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "(Song,support)": "<int>%",
            "(Song,oppose)": "<int>%",
            "(Singer,support)": "<int>%",
            "(Singer,oppose)": "<int>%",
            "(Lyrics,support)": "<int>%",
            "(Lyrics,oppose)": "<int>%",
            "(Visual,support)": "<int>%",
            "(Visual,oppose)": "<int>%"
        }}
    }}

    Answer:
"""


# ---------- P(S∣T): Stance given target ----------
EST_PRIOR_S_cond_T_TEMPLATE = """
    Question:
    Among viewers who primarily focus on the {target} of the music video "{title}", predict how their stances would distribute.
    Ensure the percentages sum to 100 and use integers only.

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "support": "<int>%",
            "oppose": "<int>%"
        }}
    }}

    Answer:
"""


# ---------- P(T∣S): Target given stance ----------
EST_PRIOR_T_cond_S_TEMPLATE = """
    Question:
    Among viewers who express a {stance_label} attitude toward the music video "{title}", predict how their discussed targets would distribute.
    Ensure the percentages sum to 100 and use integers only.

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "Song": "<int>%",
            "Singer": "<int>%",
            "Lyrics": "<int>%",
            "Visual": "<int>%"
        }}
    }}

    Answer:
"""









# Ambiguous templates
AMBIGUOUS_SYS_TEMPLATE = """
    You are an expert in analyzing public opinions about a specific music video based on the given YouTube viewer comments.
    You will be asked opinion questions about the viewer.
    Answer the question base on the comments.

    Music Video Information:
    {meta_data}

    YouTube Viewer Comments:
    {comments}
"""

# ---------- P(S): Marginal stance distribution ----------
AMBIGUOUS_S_TEMPLATE = """
    Question:
    How do most viewers feel about the music video "{title}"?
    Support or Oppose?

    Output the following JSON format:
    {{
        "stance": "<support or oppose>"
    }}

    Answer:
"""

# ---------- P(T): Marginal target distribution ----------
AMBIGUOUS_T_TEMPLATE = """
    Question:
    What do viewers discuss most when talking about the music video "{title}"?
    Song, Singer, Lyrics, or Visual?

    Output the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""

# ---------- P(S,T): Joint stance–target distribution ----------
AMBIGUOUS_T_S_TEMPLATE = """
    Question:
    Which part of the music video "{title}" receives the most praise, and which part receives the most criticism?
    Choose from Song, Singer, Lyrics or Visual.

    Output the following JSON format:
    {{
        "most_praised": "<one of the above aspects>",
        "most_criticized": "<one of the above aspects>"
    }}

    Answer:
"""


# ---------- P(S∣T): Stance given target ----------
AMBIGUOUS_S_cond_T_TEMPLATE = """
    Question:
    What does a viewer who cares about {target} of music video would feel about "{title}"?
    Support or Oppose?

    Output the following JSON format:
    {{
        "stance": "<support or oppose>"
    }}

    Answer:
"""

# ---------- P(T∣S): Target given stance ----------
AMBIGUOUS_T_cond_S_TEMPLATE = """
    Question:
    Which aspects of the music video "{title}" are most appreciated by {stance_label} viewers?
    Song, Singer, Lyrics, or Visual?

    Output the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""





# Ambiguous Prior templates
AMBIGUOUS_PRIOR_SYS_TEMPLATE = """
    You are an expert in predicting YouTube viewer opinions about a specific music video based on its information.
    You will be asked to provide predictions for several opinion-related questions.

    Music Video Information:
    {meta_data}
"""

# ---------- P(S): Marginal stance distribution ----------
AMBIGUOUS_PRIOR_S_TEMPLATE = """
    Question:
    How do you think most viewers will feel about the music video "{title}"?
    Support or Oppose?

    Output the following JSON format:
    {{
        "stance": "<support or oppose>"
    }}

    Answer:
"""

# ---------- P(T): Marginal target distribution ----------
AMBIGUOUS_PRIOR_T_TEMPLATE = """
    Question:
    What do you think viewers will talk about most when discussing the music video "{title}"?
    Song, Singer, Lyrics, or Visual?

    Output the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""

# ---------- P(S,T): Joint stance–target distribution ----------
AMBIGUOUS_PRIOR_T_S_TEMPLATE = """
    Question:
    Which part of the music video "{title}" do you think will receive the most praise, and which part will receive the most criticism?
    Choose from Song, Singer, Lyrics, or Visual?

    Output the following JSON format:
    {{
        "most_praised": "<one of the above aspects>",
        "most_criticized": "<one of the above aspects>"
    }}

    Answer:
"""

# ---------- P(S∣T): Stance given target ----------
AMBIGUOUS_PRIOR_S_cond_T_TEMPLATE = """
    Question:
    What do you think a viewer who cares about {target} of music video will feel about "{title}"?
    Support or Oppose?

    Output the following JSON format:
    {{
        "stance": "<support or oppose>"
    }}

    Answer:
"""

# ---------- P(T∣S): Target given stance ----------
AMBIGUOUS_PRIOR_T_cond_S_TEMPLATE = """
    Question:
    Which aspects of the music video "{title}" do you think will be most appreciated by {stance_label} viewers?
    Song, Singer, Lyrics, or Visual?
    
    Output the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""
