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




# QA TEMPLATES
TARGETS = ["Actor", "Storyline", "Visual", "Audio"]

# Distribution Estimation templates
EST_SYS_TEMPLATE = """
    You are an expert in analyzing public opinions about a specific movie based on the given YouTube viewer comments.

    Each comment can be interpreted along two dimensions:
    - **Stance** — the expressed attitude toward the movie or its aspects.
    - **Target** — the specific aspect or attribute of the movie being discussed.

    Stances include:
    - support: positive or approving opinions.
    - oppose: negative or critical opinions.

    Targets include:
    - Actor: performances, delivery, emotion, chemistry, casting.
    - Storyline: plot, narrative, themes, pacing, character arcs, dialogue.
    - Visual: cinematography, animation, lighting, color, production design, costumes, visual effects.
    - Audio: soundtrack, score, songs, sound effects, or audio mix.

    You will be asked to estimate quantitative distributions of stance and target using the provided information.

    Movie Information:
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
    Estimate how YouTube viewers’ comments distribute across different stances toward the movie "{title}".
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
    Estimate how YouTube viewers’ comments distribute across the different targets discussed in the movie "{title}".
    Ensure the percentages sum to 100 and use integers only.

    {hints}

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "Actor": "<int>%",
            "Storyline": "<int>%",
            "Visual": "<int>%",
            "Audio": "<int>%"
        }}
    }}

    Answer:
"""

# ---------- P(S,T): Joint stance–target distribution ----------
EST_T_S_TEMPLATE = """
    Question:
    Estimate the joint distribution of stances and targets among YouTube comments for the movie "{title}".
    Ensure the percentages sum to 100 and use integers only.

    {hints}

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "(Actor,support)": "<int>%",
            "(Actor,oppose)": "<int>%",
            "(Storyline,support)": "<int>%",
            "(Storyline,oppose)": "<int>%",
            "(Visual,support)": "<int>%",
            "(Visual,oppose)": "<int>%",
            "(Audio,support)": "<int>%",
            "(Audio,oppose)": "<int>%"
        }}
    }}

    Answer:
"""

# ---------- P(S∣T): Stance given target ----------
EST_S_cond_T_TEMPLATE = """
    Question:
    Among comments discussing the {target} in the movie "{title}", estimate how viewers’ stances distribute.
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
    Among {stance_label} comments toward the movie "{title}", estimate how these comments distribute across different targets.
    Ensure the percentages sum to 100 and use integers only.

    {hints}

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "Actor": "<int>%",
            "Storyline": "<int>%",
            "Visual": "<int>%",
            "Audio": "<int>%"
        }}
    }}

    Answer:
"""



# Prior Predict templates
PRIOR_SYS_TEMPLATE = """
    You are an expert in predicting YouTube viewer opinions about a specific movie based on its information.

    Each viewer’s opinion can be interpreted along two dimensions:
    - **Stance** — the expressed attitude toward the movie or its aspects.
    - **Target** — the specific aspect or attribute of the movie being discussed.

    Stances include:
    - "support": positive or approving opinions.
    - "oppose": negative or critical opinions.

    Targets include:
    - "Actor": performances, delivery, emotion, chemistry, casting.
    - "Storyline": plot, narrative, themes, pacing, character arcs, dialogue.
    - "Visual": cinematography, animation, lighting, color, production design, costumes, visual effects.
    - "Audio": soundtrack, score, songs, sound effects, or audio mix.

    You will be asked to predict quantitative distributions of stance and target using the movie information.

    Movie Information:
    {meta_data}
"""

# ---------- P(S): Marginal stance distribution ----------
PRIOR_S_TEMPLATE = """
    Question:
    Predict how YouTube viewers would distribute across different stances toward the movie "{title}".
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
PRIOR_T_TEMPLATE = """
    Question:
    Predict how YouTube viewers would distribute across the different targets discussed in the movie "{title}".
    Ensure the percentages sum to 100 and use integers only.

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "Actor": "<int>%",
            "Storyline": "<int>%",
            "Visual": "<int>%",
            "Audio": "<int>%"
        }}
    }}

    Answer:
"""

# ---------- P(S,T): Joint stance–target distribution ----------
PRIOR_T_S_TEMPLATE = """
    Question:
    Predict the joint distribution of stances and targets among YouTube viewers for the movie "{title}".
    Ensure the percentages sum to 100 and use integers only.

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "(Actor,support)": "<int>%",
            "(Actor,oppose)": "<int>%",
            "(Storyline,support)": "<int>%",
            "(Storyline,oppose)": "<int>%",
            "(Visual,support)": "<int>%",
            "(Visual,oppose)": "<int>%",
            "(Audio,support)": "<int>%",
            "(Audio,oppose)": "<int>%"
        }}
    }}

    Answer:
"""

# ---------- P(S∣T): Stance given target ----------
PRIOR_S_cond_T_TEMPLATE = """
    Question:
    Among viewers who primarily focus on the {target} of the movie "{title}", predict how their stances would distribute.
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
PRIOR_T_cond_S_TEMPLATE = """
    Question:
    Among viewers who express a {stance_label} attitude toward the movie "{title}", predict how their discussed targets would distribute.
    Ensure the percentages sum to 100 and use integers only.

    Output distribution in the following JSON format:
    {{
        "percentages": {{
            "Actor": "<int>%",
            "Storyline": "<int>%",
            "Visual": "<int>%",
            "Audio": "<int>%"
        }}
    }}

    Answer:
"""




# Ambiguous templates
AMBIGUOUS_SYS_TEMPLATE = """
    You are an expert in analyzing public opinions about a specific movie based on the given YouTube viewer comments.
    You will be asked opinion questions about the viewer.
    Answer the question base on the comments.

    Movie Information:
    {meta_data}

    YouTube Viewer Comments:
    {comments}
"""

# ---------- P(S): Marginal stance distribution ----------
AMBIGUOUS_S_TEMPLATE = """
    Question:
    How do most viewers feel about the movie "{title}"?
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
    What do viewers discuss most when talking about the movie "{title}"?
    Actor, Storyline, Visual, Audio?

    Output the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""

# ---------- P(S,T): Joint stance–target distribution ----------
AMBIGUOUS_T_S_TEMPLATE = """
    Question:
    Which part of the movie "{title}" receives the most praise, and which part receives the most criticism?
    Choose from Actor, Storyline, Visual, or Audio.

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
    What does a viewer who cares about {target} of the movie would feel about "{title}"?
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
    Which aspects of the movie "{title}" are most appreciated by {stance_label} viewers?
    Actor, Storyline, Visual, or Audio?

    Output the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""





# Ambiguous Prior templates
AMBIGUOUS_PRIOR_SYS_TEMPLATE = """
    You are an expert in predicting YouTube viewer opinions about a specific movie based on its information.
    You will be asked to provide predictions for several opinion-related questions.

    Movie Information:
    {meta_data}
"""

# ---------- P(S): Marginal stance distribution ----------
AMBIGUOUS_PRIOR_S_TEMPLATE = """
    Question:
    How do you think most viewers will feel about the movie "{title}"?
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
    What do you think viewers will talk about most when discussing the movie "{title}"?
    Actor, Storyline, Visual, or Audio?

    Output the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""

# ---------- P(S,T): Joint stance–target distribution ----------
AMBIGUOUS_PRIOR_T_S_TEMPLATE = """
    Question:
    Which part of the movie "{title}" do you think will receive the most praise, and which part will receive the most criticism?
    Choose from Actor, Storyline, Visual, or Audio.

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
    What do you think a viewer who cares about {target} of movie will feel about "{title}"?
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
    Which aspects of the movie "{title}" do you think will be most appreciated by {stance_label} viewers?
    Actor, Storyline, Visual, or Audio?
    
    Output the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""
