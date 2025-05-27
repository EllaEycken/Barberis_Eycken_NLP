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

"""IMPORT STATEMENTS"""

import statistics
import math
from parselmouth.praat import call  # if installing parselmouth: pip install praat-parselmouth

import epitran
import nltk
import pandas as pd
import spacy

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (  # TODO: if all is finished, switch this to AUDIO_DIR!
    AUDIO_DIR_DUMMY,
    period_ceiling,
    unit,
)
from masterthesisellalaw.features.helper_functions.helper_functions_praat import (
    calculate_duration,
    calculate_durationsOfPeriodsLists,
    calculate_meanIntensity,
    calculate_syllNucleiFeinberg,
    create_new_sound_only_sounding,
    create_pitchObject,
    create_pointProcessObject,
    create_textGridDataframe,
    create_textGridSilencesObject,
    create_textGridVuvNoSilencesObject,
    extract_partsOfSound,
)
from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import (
    read_sounds,
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


""" FLUENCY """



"""RUNNING THE FUNCTIONS"""
if __name__ == "__main__":
    text_dir = TEXT_DIR_DUMMY