# QA TEMPLATES
TARGETS = ["Actor", "Storyline", "Visual", "Audio"]


SYS_TEMPLATE = """
    You will be given information about a movie.

    Based on this information, consider how viewers are likely to react to the movie and what aspects they are likely to talk about.

    When thinking about viewer opinions, keep in mind two dimensions:
    - Stance: whether viewers are likely to express a positive or negative attitude.
    - Topic: which aspect of the movie viewers are likely to focus on.

    Stance categories:
    - positive: expressing approval, enjoyment, or praise.
    - negative: expressing criticism, dissatisfaction, or disappointment.

    Topic categories:
    - Actor: acting performance, casting, chemistry, emotional expression.
    - Storyline: plot, narrative, pacing, themes, dialogue, character development.
    - Visual: cinematography, animation, lighting, color, visual effects, production design.
    - Audio: soundtrack, music, sound effects, or audio quality.

    Read the movie information carefully.
    Then answer the question based on this information.

    Movie Information:
    {meta_data}
"""



# ---------- P(S): Marginal stance distribution ----------
EST_S_TEMPLATE = """
Question:
Read the movie information.
How are viewers likely to feel overall about the movie?
Predict how common each attitude is expected to be.
Ensure the percentages sum to 100 and use integers only.

Output your prediction in the following JSON format:
{{
    "percentages": {{
        "positive": "<int>%",
        "negative": "<int>%"
    }}
}}

Answer:
"""


# ---------- P(T): Marginal topic distribution ----------
EST_T_TEMPLATE = """
Question:
Read the movie information.
Which aspects of the movie are viewers most likely to talk about?
Predict how frequently each aspect is expected to be discussed.
Ensure the percentages sum to 100 and use integers only.

Output your prediction in the following JSON format:
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


# ---------- P(S,T): Joint stance–topic distribution ----------
EST_T_S_TEMPLATE = """
Question:
Read the movie information.
Consider both which aspect viewers are likely to discuss and whether their opinions are likely to be positive or negative.
Predict how these combinations are expected to appear.
Ensure the percentages sum to 100 and use integers only.

Output your prediction in the following JSON format:
{{
    "percentages": {{
        "(Actor,positive)": "<int>%",
        "(Actor,negative)": "<int>%",
        "(Storyline,positive)": "<int>%",
        "(Storyline,negative)": "<int>%",
        "(Visual,positive)": "<int>%",
        "(Visual,negative)": "<int>%",
        "(Audio,positive)": "<int>%",
        "(Audio,negative)": "<int>%"
    }}
}}

Answer:
"""


# ---------- P(S∣T): Stance given topic ----------
EST_S_cond_T_TEMPLATE = """
Question:
Read the movie information.
Focus on viewers who are likely to care most about the {topic} aspect of the movie.
How are their attitudes toward the movie likely to be divided?
Ensure the percentages sum to 100 and use integers only.

Output your prediction in the following JSON format:
{{
    "percentages": {{
        "positive": "<int>%",
        "negative": "<int>%"
    }}
}}

Answer:
"""


# ---------- P(T∣S): Target given stance ----------
EST_T_cond_S_TEMPLATE = """
Question:
Read the movie information. 
Focus on viewers who are likely to have a {stance_label} attitude toward the movie.
Which aspects of the movie are these viewers most likely to talk about?
Ensure the percentages sum to 100 and use integers only.

Output your prediction in the following JSON format:
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



MOSTFREQ_S_TEMPLATE = """
Question:
Read the movie information. 
What overall attitude are most viewers likely to have?

Output your prediction in the following JSON format:
{{
    "stance": "<positive or negative>"
}}

Answer:
"""

MOSTFREQ_T_TEMPLATE = """
Question:
Read the movie information. 
Which aspect of the movie are viewers most likely to discuss?
Choose from: Actor, Storyline, Visual, or Audio.

Output your prediction in the following JSON format:
{{
    "aspect": "<one of the above aspects>"
}}

Answer:
"""



MOSTFREQ_T_S_TEMPLATE = """
    Question:
    Read the movie information. 
    Considering both (1) which aspect is being talked about and (2) whether the attitude is positive or negative.
    Which combination appears most likely overall?

    Choose one pair from:
    (Actor,positive), (Actor,negative),
    (Storyline,positive), (Storyline,negative),
    (Visual,positive), (Visual,negative),
    (Audio,positive), (Audio,negative).

    Output your prediction in the following JSON format:
    {{
        "combination": "<one of the combinations above>"
    }}

    Answer:
"""


MOSTFREQ_S_cond_T_TEMPLATE = """
Question:
Read the movie information. 
Focus on viewers who are likely to care most about the {topic} aspect of the movie.
What attitude are these viewers most likely to express?

Output your prediction in the following JSON format:
{{
    "stance": "<positive or negative>"
}}

Answer:
"""


MOSTFREQ_T_cond_S_TEMPLATE = """
Question:
Read the movie information. 
Focus on viewers who are likely to have a {stance_label} attitude toward the movie.
Which aspect of the movie are they most likely to talk about?
Choose from: Actor, Storyline, Visual, or Audio.

Output your prediction in the following JSON format:
{{
    "aspect": "<one of the above aspects>"
}}

Answer:
"""



# ---------- SECONDMOST ---------- #
SECONDMOST_S_TEMPLATE = """
    Question:
    Read the movie information. 
    What overall attitude are viewers likely to have as the second most common?

    Output your prediction in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""

SECONDMOST_T_TEMPLATE = """
    Question:
    Read the movie information. 
    Which aspect of the movie are viewers likely to discuss as the second most common?
    Choose from: Actor, Storyline, Visual, or Audio.

    Output your prediction in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""


SECONDMOST_T_S_TEMPLATE = """
    Question:
    Read the movie information. 
    Considering both (1) which aspect is being talked about and (2) whether the attitude is positive or negative.
    Which combination is the second most likely overall?

    Choose one pair from:
    (Actor,positive), (Actor,negative),
    (Storyline,positive), (Storyline,negative),
    (Visual,positive), (Visual,negative),
    (Audio,positive), (Audio,negative).

    Output your prediction in the following JSON format:
    {{
        "combination": "<one of the combinations above>"
    }}

    Answer:
"""


SECONDMOST_S_cond_T_TEMPLATE = """
    Question:
    Read the movie information. 
    Focus on viewers who are likely to care most about the {topic} aspect of the movie.
    What attitude are these viewers likely to express as the second most common?

    Output your prediction in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""

SECONDMOST_T_cond_S_TEMPLATE = """
    Question:
    Read the movie information. 
    Focus on viewers who are likely to have a {stance_label} attitude toward the movie.
    Which aspect of the movie are they likely to talk about as the second most common?
    Choose from: Actor, Storyline, Visual, or Audio.

    Output your prediction in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""


