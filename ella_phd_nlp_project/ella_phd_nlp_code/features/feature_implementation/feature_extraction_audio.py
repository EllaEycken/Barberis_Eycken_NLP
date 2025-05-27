"""Extract linguistic features from audio-files.
The following features are extracted:
- Fluency
    - Speech rate
    - Short Pause
    - Long Pause

Note: Fluency measures extracted from TEXT files (see feature_extraction_text.py) include the following:
    - Filled Pause
    - False Start
    - Word Abandoned
    - Word Repetition

"""

import statistics
import math

import epitran
import nltk
import pandas as pd
import spacy

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    # AoA_PATH,
    # CONCRETENESS_PATH,
    # FREQUENCY_PATH,
    # DUTCH_ALPHABET,
    # FAMILIARITY_PATH,
    # MILTENBURG_MODEL_PATH,
    # NAME_AGREEMENT_PATH,
    TEXT_DIR,
)

# from masterthesisellalaw.data.preprocess_norms import (
# read_Norms_AoA,
# read_Norms_concreteness,
# read_Norms_familiarity,
# read_Norms_frequency,
# read_Norms_nameAgreement,
# )

from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import (
    read_transcripts,
)

epi = epitran.Epitran("nld-Latn")
nlp = spacy.load("nl_core_news_lg")

# linguistic features: language --> Lexical features
# List:
# - Lexical Variation
# - Word finding problems/lexical recovery
# - Occurrence Frequency/Lexical Content
# - Psycholinguistic word variables/lexical content
# - Lexical Access
# - Complexity
# - Vocabulary Size

# IDEA: use the Dutch tagger from EVANMILTENBURG
# https://github.com/evanmiltenburg/Dutch-tagger/tree/master
# This may take a few minutes. (But once loaded, the tagger is really fast!)
tagger = PerceptronTagger(load=False)
tagger.load(MILTENBURG_MODEL_PATH)
