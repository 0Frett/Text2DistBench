# QA TEMPLATES
TARGETS = ["Song", "Singer", "Visual", "Lyrics"]


SYS_TEMPLATE = """
    You will be given information about a music video.

    Based on this information, consider how viewers are likely to react to the music video and what aspects they are likely to talk about.

    When thinking about viewer opinions, keep in mind two dimensions:
    - Stance: whether viewers are likely to express a positive or negative attitude.
    - Topic: which aspect of the music video viewers are likely to focus on.

    Stance categories:
    - positive: expressing approval, enjoyment, or praise.
    - negative: expressing criticism, dissatisfaction, or disappointment.

    Topic categories:
    - Song: musical composition, melody, harmony, rhythm, structure (verse/chorus/bridge), or arrangement.
    - Singer: vocal tone, technique (runs/belts/falsetto), emotion, or performance.
    - Lyrics: words and meaning, themes, storytelling, message, or rhyme.
    - Visual: cinematography, lighting, color, concept, choreography, or animation/VFX.

    Read the music video information carefully.
    Then answer the question based on this information.

    Music Video Information:
    {meta_data}
"""



# ---------- P(S): Marginal stance distribution ----------
EST_S_TEMPLATE = """
Question:
Read the music video information.
How are viewers likely to feel overall about the music video?
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
Read the music video information.
Which aspects of the music video are viewers most likely to talk about?
Predict how frequently each aspect is expected to be discussed.
Ensure the percentages sum to 100 and use integers only.

Output your prediction in the following JSON format:
{{
    "percentages": {{
        "Song": "<int>%",
        "Singer": "<int>%",
        "Visual": "<int>%",
        "Lyrics": "<int>%"
    }}
}}

Answer:
"""


# ---------- P(S,T): Joint stance–topic distribution ----------
EST_T_S_TEMPLATE = """
Question:
Read the music video information.
Consider both which aspect viewers are likely to discuss and whether their opinions are likely to be positive or negative.
Predict how these combinations are expected to appear.
Ensure the percentages sum to 100 and use integers only.

Output your prediction in the following JSON format:
{{
    "percentages": {{
        "(Song,positive)": "<int>%",
        "(Song,negative)": "<int>%",
        "(Singer,positive)": "<int>%",
        "(Singer,negative)": "<int>%",
        "(Visual,positive)": "<int>%",
        "(Visual,negative)": "<int>%",
        "(Lyrics,positive)": "<int>%",
        "(Lyrics,negative)": "<int>%"
    }}
}}

Answer:
"""


# ---------- P(S∣T): Stance given topic ----------
EST_S_cond_T_TEMPLATE = """
Question:
Read the music video information.
Focus on viewers who are likely to care most about the {topic} aspect of the music video.
How are their attitudes toward the music video likely to be divided?
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
Read the music video information. 
Focus on viewers who are likely to have a {stance_label} attitude toward the music video.
Which aspects of the music video are these viewers most likely to talk about?
Ensure the percentages sum to 100 and use integers only.

Output your prediction in the following JSON format:
{{
    "percentages": {{
        "Song": "<int>%",
        "Singer": "<int>%",
        "Visual": "<int>%",
        "Lyrics": "<int>%"
    }}
}}

Answer:
"""



MOSTFREQ_S_TEMPLATE = """
Question:
Read the music video information. 
What overall attitude are most viewers likely to have?

Output your prediction in the following JSON format:
{{
    "stance": "<positive or negative>"
}}

Answer:
"""

MOSTFREQ_T_TEMPLATE = """
Question:
Read the music video information. 
Which aspect of the music video are viewers most likely to discuss?
Choose from: Song, Singer, Visual, or Lyrics.

Output your prediction in the following JSON format:
{{
    "aspect": "<one of the above aspects>"
}}

Answer:
"""

MOSTFREQ_T_S_TEMPLATE = """
Question:
Read the music video information. 
Considering both (1) which aspect is being talked about and (2) whether the attitude is positive or negative.
Which combination appears most likely overall?

Choose one pair from:
(Song,positive), (Song,negative),
(Singer,positive), (Singer,negative),
(Visual,positive), (Visual,negative),
(Lyrics,positive), (Lyrics,negative).

Output your prediction in the following JSON format:
{{
    "combination": "<one of the combinations above>"
}}

Answer:
"""

MOSTFREQ_S_cond_T_TEMPLATE = """
Question:
Read the music video information. 
Focus on viewers who are likely to care most about the {topic} aspect of the music video.
What attitude are these viewers most likely to express?

Output your prediction in the following JSON format:
{{
    "stance": "<positive or negative>"
}}

Answer:
"""


MOSTFREQ_T_cond_S_TEMPLATE = """
Question:
Read the music video information. 
Focus on viewers who are likely to have a {stance_label} attitude toward the music video.
Which aspect of the music video are they most likely to talk about?
Choose from: Song, Singer, Visual, or Lyrics.

Output your prediction in the following JSON format:
{{
    "aspect": "<one of the above aspects>"
}}

Answer:
"""


LEASTFREQ_S_TEMPLATE = """
Question:
Read the music video information. 
Which overall attitude is likely to be the least common among viewers?

Output your prediction in the following JSON format:
{{
    "stance": "<positive or negative>"
}}

Answer:
"""

LEASTFREQ_T_TEMPLATE = """
Question:
Read the music video information. 
Which aspect of the music video is viewers least likely to talk about?
Choose from: Song, Singer, Visual, or Lyrics.

Output your prediction in the following JSON format:
{{
    "aspect": "<one of the above aspects>"
}}

Answer:
"""

LEASTFREQ_T_S_TEMPLATE = """
    Question:
    Read the music video information. 
    Considering both (1) which aspect is being talked about and (2) whether the attitude is positive or negative.
    Which combination is least likely overall?

    Choose one pair from:
    (Song,positive), (Song,negative),
    (Singer,positive), (Singer,negative),
    (Visual,positive), (Visual,negative),
    (Lyrics,positive), (Lyrics,negative).

    Output your prediction in the following JSON format:
    {{
        "combination": "<one of the combinations above>"
    }}

    Answer:
"""


LEASTFREQ_S_cond_T_TEMPLATE = """
    Question:
    Read the music video information.  
    Focus on viewers who are likely to care most about the {topic} aspect of the music video.
    Which attitude is least likely among these viewers?

    Output your prediction in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""

LEASTFREQ_T_cond_S_TEMPLATE = """
    Question:
    Read the music video information.  
    Focus on viewers who are likely to have a {stance_label} attitude toward the music video.
    Which aspect of the music video are they least likely to mention?
    Choose from: Song, Singer, Visual, or Lyrics.

    Output your prediction in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""



# ---------- SECONDMOST ---------- #
SECONDMOST_S_TEMPLATE = """
    Question:
    Read the music video information. 
    What overall attitude are viewers likely to have as the second most common?

    Output your prediction in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""

SECONDMOST_T_TEMPLATE = """
    Question:
    Read the music video information. 
    Which aspect of the music video are viewers likely to discuss as the second most common?
    Choose from: Song, Singer, Visual, or Lyrics.

    Output your prediction in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""


SECONDMOST_T_S_TEMPLATE = """
    Question:
    Read the music video information. 
    Considering both (1) which aspect is being talked about and (2) whether the attitude is positive or negative.
    Which combination is the second most likely overall?

    Choose one pair from:
    (Song,positive), (Song,negative),
    (Singer,positive), (Singer,negative),
    (Visual,positive), (Visual,negative),
    (Lyrics,positive), (Lyrics,negative).

    Output your prediction in the following JSON format:
    {{
        "combination": "<one of the combinations above>"
    }}

    Answer:
"""


SECONDMOST_S_cond_T_TEMPLATE = """
    Question:
    Read the music video information. 
    Focus on viewers who are likely to care most about the {topic} aspect of the music video.
    What attitude are these viewers likely to express as the second most common?

    Output your prediction in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""


SECONDMOST_T_cond_S_TEMPLATE = """
    Question:
    Read the music video information. 
    Focus on viewers who are likely to have a {stance_label} attitude toward the music video.
    Which aspect of the music video are they likely to talk about as the second most common?
    Choose from: Song, Singer, Visual, or Lyrics.

    Output your prediction in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""
