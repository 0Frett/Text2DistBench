
# QA TEMPLATES
TARGETS = ["Song", "Singer", "Visual", "Lyrics"]


SYS_TEMPLATE = """
    You will be given information about a music video, followed by a collection of viewer comments.
    Each comment reflects what a viewer thinks about the music video and focuses on a particular aspect while expressing a certain attitude.

    When reading the comments, keep in mind two dimensions:
    - Stance: whether the comment expresses a positive or negative attitude.
    - Topic: which aspect of the music video the comment mainly talks about.

    Stance categories:
    - positive: expressing approval, enjoyment, or praise.
    - negative: expressing criticism, dissatisfaction, or disappointment.

    Topic categories:
    - Song: musical composition, melody, harmony, rhythm, structure (verse/chorus/bridge), or arrangement.
    - Singer: vocal tone, technique (runs/belts/falsetto), emotion, or performance.
    - Lyrics: words and meaning, themes, storytelling, message, or rhyme.
    - Visual: cinematography, lighting, color, concept, choreography, or animation/VFX.

    Read the following music video information and the viewer comments carefully.
    Then answer the question based on this information.

    Music Video Information:
    {meta_data}

    Viewer Comments:
    {comments}
"""


# ---------- P(S): Marginal stance distribution ----------
EST_S_TEMPLATE = """
    Question:
    Read the viewer comments. 
    How do the expressed opinions break down in terms of overall attitude toward the music video?
    Summarize how common each stance is among the comments.
    Ensure the percentages sum to 100 and use integers only.

    Output your answer in the following JSON format:
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
    Read the viewer comments. 
    What aspects of the music video do viewers talk about?
    Summarize how frequently each aspect appears in the comments.
    Ensure the percentages sum to 100 and use integers only.

    Output your answer in the following JSON format:
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
    Read the viewer comments. 
    Consider both which aspect is being discussed and whether the expressed opinion is positive or negative.
    Summarize how these combinations appear in the comments.
    Ensure the percentages sum to 100 and use integers only.

    Output your answer in the following JSON format:
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
    Read the viewer comments. 
    Focus only on the comments that talk about the {topic} aspect of the music video.
    How are viewers’ attitudes divided?
    Ensure the percentages sum to 100 and use integers only.

    Output your answer in the following JSON format:
    {{
        "percentages": {{
            "positive": "<int>%",
            "negative": "<int>%"
        }}
    }}

    Answer:
"""


# ---------- P(T∣S): Topic given stance ----------
EST_T_cond_S_TEMPLATE = """
    Question:
    Read the viewer comments. 
    Focus only on the comments that express a {stance_label} attitude toward the music video.
    Which aspects of the music video do these comments discuss?
    Ensure the percentages sum to 100 and use integers only.

    Output your answer in the following JSON format:
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
    Read the viewer comments.  
    What overall attitude do most viewers express?

    Output your answer in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""

MOSTFREQ_T_TEMPLATE = """
    Question:
    Read the viewer comments. 
    Which aspect of the music video is discussed most often?
    Choose from: Song, Singer, Visual, or Lyrics.

    Output your answer in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""


MOSTFREQ_T_S_TEMPLATE = """
Question:
Read the viewer comments. 
Considering both (1) which aspect is being talked about and (2) whether the attitude is positive or negative.
Which combination appears most often in the comments?

Choose one pair from:
(Song,positive), (Song,negative),
(Singer,positive), (Singer,negative),
(Visual,positive), (Visual,negative),
(Lyrics,positive), (Lyrics,negative).

Output your answer in the following JSON format:
{{
    "combination": "<one of the combinations above>"
}}

Answer:
"""


MOSTFREQ_S_cond_T_TEMPLATE = """
    Question:
    Read the viewer comments.
    Focus only on the comments that talk about the {topic} aspect of the music video.
    What attitude do viewers most commonly express?

    Output your answer in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""


MOSTFREQ_T_cond_S_TEMPLATE = """
    Question:
    Read the viewer comments.
    Focus only on the comments that express a {stance_label} attitude toward the music video.
    Which aspect of the music video is mentioned most often?
    Choose from: Song, Singer, Visual, or Lyrics.

    Output your answer in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""


LEASTFREQ_S_TEMPLATE = """
    Question:
    Read the viewer comments.
    Which overall attitude appears least often?

    Output your answer in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""

LEASTFREQ_T_TEMPLATE = """
    Question:
    Read the viewer comments. 
    Which aspect of the music video is mentioned the least?
    Choose from: Song, Singer, Visual, or Lyrics.

    Output your answer in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""


LEASTFREQ_T_S_TEMPLATE = """
    Question:
    Read the viewer comments. 
    Considering both (1) which aspect is being talked about and (2) whether the attitude is positive or negative.
    Which combination appears least often in the comments?

    Choose one pair from:
    (Song,positive), (Song,negative),
    (Singer,positive), (Singer,negative),
    (Visual,positive), (Visual,negative),
    (Lyrics,positive), (Lyrics,negative).

    Output your answer in the following JSON format:
    {{
        "combination": "<one of the combinations above>"
    }}

    Answer:
"""


LEASTFREQ_S_cond_T_TEMPLATE = """
    Question:
    Read the viewer comments. 
    Focus only on the comments that talk about the {topic} aspect of the music video.
    Which attitude appears least often?

    Output your answer in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""


LEASTFREQ_T_cond_S_TEMPLATE = """
    Question:
    Read the viewer comments.
    Focus only on the comments that express a {stance_label} attitude toward the music video.
    Which aspect of the music video is mentioned the least?
    Choose from: Song, Singer, Visual, or Lyrics.

    Output your answer in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""



# ---------- SECONDMOST ---------- #
SECONDMOST_S_TEMPLATE = """
    Question:
    Read the viewer comments.  
    What overall attitude is the second most commonly expressed by viewers?

    Output your answer in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""

SECONDMOST_T_TEMPLATE = """
    Question:
    Read the viewer comments. 
    Which aspect of the music video is discussed the second most often?
    Choose from: Song, Singer, Visual, or Lyrics.

    Output your answer in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""


SECONDMOST_T_S_TEMPLATE = """
    Question:
    Read the viewer comments. 
    Considering both (1) which aspect is being talked about and (2) whether the attitude is positive or negative.
    Which combination appears the second most often in the comments?

    Choose one pair from:
    (Song,positive), (Song,negative),
    (Singer,positive), (Singer,negative),
    (Visual,positive), (Visual,negative),
    (Lyrics,positive), (Lyrics,negative).

    Output your answer in the following JSON format:
    {{
        "combination": "<one of the combinations above>"
    }}

    Answer:
"""


SECONDMOST_S_cond_T_TEMPLATE = """
    Question:
    Read the viewer comments.
    Focus only on the comments that talk about the {topic} aspect of the music video.
    What attitude is the second most commonly expressed?

    Output your answer in the following JSON format:
    {{
        "stance": "<positive or negative>"
    }}

    Answer:
"""


SECONDMOST_T_cond_S_TEMPLATE = """
    Question:
    Read the viewer comments.
    Focus only on the comments that express a {stance_label} attitude toward the music video.
    Which aspect of the music video is mentioned the second most often?
    Choose from: Song, Singer, Visual, or Lyrics.

    Output your answer in the following JSON format:
    {{
        "aspect": "<one of the above aspects>"
    }}

    Answer:
"""