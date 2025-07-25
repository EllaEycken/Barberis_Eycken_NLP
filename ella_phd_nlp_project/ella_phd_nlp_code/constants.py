""" Constants for the project. """

# Imports
import os

## Load environment variables
from dotenv import load_dotenv
# This line imports the load_dotenv function from the dotenv package. The dotenv package is commonly used to load
# environment variables from a .env file into the environment, making it easier to manage configuration settings.

load_dotenv()
# This function call loads the environment variables defined in a .env file located in the same directory as the script
# or in the root directory of the project. After this line is executed, you can access these variables using os.getenv()
# or os.environ.

HERE = os.path.dirname(os.path.realpath(__file__))
# This line retrieves the absolute path of the directory where the current script (__file__) is located.
# os.path.realpath(__file__) returns the canonical path of the script, and os.path.dirname() gets the directory part
# of that path. The result is stored in the variable HERE.

PROJECT_DIR = os.path.join(os.path.abspath(os.path.join(HERE, os.pardir)))
# This line constructs the absolute path to the parent directory of the current script's directory.
# os.pardir is a constant that represents the parent directory (i.e., ..).
# os.path.join(HERE, os.pardir) combines the HERE path with the parent directory, and os.path.abspath()
# ensures it resolves to an absolute path.
# The result is stored in the variable PROJECT_DIR.


## Directories for DATA
DATA_DIR = os.path.join(PROJECT_DIR, '../data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

DOCX_DIR = os.path.join(RAW_DIR, 'transcripts')
DOCX_DIR_DUMMY = os.path.join(RAW_DIR, 'transcripts_dummy')
DEMOGRAPHICS_DIR = os.path.join(RAW_DIR,'demographics')
PRECLEANTEXT_DIR = os.path.join(INTERIM_DIR, 'transcripts_preclean')
PRECLEANTEXT_DIR_DUMMY = os.path.join(INTERIM_DIR, 'txt_transcripts_preclean_dummy')
TEXT_DIR = os.path.join(PROCESSED_DIR, 'txt_transcripts')
TEXT_DIR_DUMMY = os.path.join(PROCESSED_DIR, 'txt_transcripts_dummy')

AUDIO_DIR = os.path.join(RAW_DIR, 'audio')
AUDIO_DIR_DUMMY = os.path.join(RAW_DIR, 'audio_dummy')
DIAR_DIR = os.path.join(RAW_DIR, 'diarization')
DIAR_DIR_DUMMY = os.path.join(RAW_DIR, 'diarization_dummy')
NONMERGED_AUDIO_PATIENT_DIR = os.path.join(INTERIM_DIR, 'audio_patientonly_nonmerged')
NONMERGED_AUDIO_PATIENT_DIR_DUMMY = os.path.join(INTERIM_DIR, 'audio_patientonly_nonmerged_dummy')
NONMERGED_AUDIO_PATIENTU_DIR = os.path.join(INTERIM_DIR, 'audio_patientonlyU_nonmerged')
NONMERGED_AUDIO_PATIENTU_DIR_DUMMY = os.path.join(INTERIM_DIR, 'audio_patientonlyU_nonmerged_dummy')
# CLEAN_DIAR_DIR = os.path.join(INTERIM_DIR, 'diarization_clean')
# CLEAN_DIAR_DIR_DUMMY = os.path.join(INTERIM_DIR, 'diarization_clean_dummy')
AUDIO_PATIENT_DIR = os.path.join(PROCESSED_DIR, 'audio_patientonly')
AUDIO_PATIENT_DIR_DUMMY = os.path.join(PROCESSED_DIR, 'audio_patientonly_dummy')
AUDIO_PATIENTU_DIR = os.path.join(PROCESSED_DIR, 'audio_patientonlyU')
AUDIO_PATIENTU_DIR_DUMMY = os.path.join(PROCESSED_DIR, 'audio_patientonlyU_dummy')

## Directory for models
MODELS_DIR = os.path.join(PROJECT_DIR, '../models')
MILTENBURG_MODEL_PATH = os.path.join(MODELS_DIR, 'miltenburg', 'model.perc.dutch_tagger_large.pickle')
# A set of non-noun words in Dutch that should not be included in certain parts of the analyses
NON_NOUNS = {
    "wij",
    "ons",
    "onze",
    "we",
    "zij",
    "ze",
    "hun",
    "hen",
    "hij",
    "hem",
    "zijn",
    "haar",
    "hare",
    "ik",
    "mij",
    "me",
    "mijn",
    "jullie",
    "jij",
    "u",
    "uw",
    "uwe",
    "je",
    "gij",
    "ge",
    "jouw",
    "jou",
    "dat",
    "da",
    "die",
    "dit",
    "deze",
    "dees",
    "het",
    "de",
    '"s',
    "r",
    "*",
    "uh",
    "eh",
}

## Directories for reports
REPORTS_DIR = os.path.join(PROJECT_DIR, '../reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')


## Path to diagnostic labels
LABEL_PATH = os.path.join(DEMOGRAPHICS_DIR, 'labels_meta.csv')  # todo: make labels_meta.csv file


## Q-file mappings
Q_FILE_MAPPINGS = {
    "q1_subject_wise": "ANTAT",  # if looking for question 1, then looking at files with 'bio' in them etc
    "q2_subject_wise": "CAT-NL",  # important: subject = participant!!
    "q3_subject_wise": "MCA",
    "q4_subject_wise": "narrative",
}


## Constants for linguistic parameters
beletselteken_utf8 = 'â€¦'  # utf_8 encodering voor '...'
accent_à_utf8 = 'Ã'  # utf_8 encodering voor à
accent_é_utf8 = 'Ã©' #utf_8 encodering voor é

modal_lemmas = {"kunnen", "moeten", "mogen", "willen", "zullen", "hoeven"}
particles_dutch = {"eigenlijk", "ja", "nee", "toch", "nou", "wel", "hoor", "he", "inderdaad", "eens",
                "even", "allez", "allee", "oei", "goh", "ze", "se", "enzo", "enzovoort", "zenne", "enfin"}
articles_dutch = {"de", "het", "een", "ne", "nen"}  # note: "ne" and "nen" are tussentaal

annot_foreign_language = '*v'  # v = vreemde taal (Dutch)
annot_neologism = '*n'
annot_aborted_word_or_sound = '*a'
annot_false_start = '*h'  # h = herneming (Dutch)
annot_unintelligible_word = '*x'
annot_filled_pause = '*g' # g = gevulde pauze (Dutch)
annots_phonemic_paraphasia = ['*F', '*f', '*Fa', '*Fo', '*Fs', '*Ft']
annots_semantic_paraphasia = ['*S', '*s']
annot_grammatic_error = '*gr'
annot_discourse_particle = '*p'
annots_dialect = ['*D', '*Dw', '*Dk']
annot_repetition = '*r'  # note: this annotation is ADDED in the function 'clean transcripts'

## Constants for acoustic parameters
f0min = 75  # for men and women
f0max = 500  # for men and women
unit = "Hertz"
# source: https://github.com/drfeinberg/PraatScripts/blob/master/Measure%20Pitch%20and%20HNR.ipynb
period_ceiling = 1.25 / f0min  # period ceiling = 1.25 divided by pitch floor aka f0min (from PRAAT)
# Source: https://www.fon.hum.uva.nl/praat/manual/Voice_2__Jitter.html
silencedB = -25
# from: https://www.researchgate.net/publication/
# 24274554_Praat_script_to_detect_syllable_nuclei_and_measure_speech_rate_automatically
mindip = 2
minpause = 0.25
minshortpause = 0.150
# Source: Short  pauses = Number of silent regions between words that are 150-400 ms
# (Le et al., 2018; Pakhomov et al., 2010) divided by the total number of words.
maxshortpause = 0.400
# Source: Short  pauses = Number of silent regions between words that are 150-400 ms
# (Le et al., 2018; Pakhomov et al., 2010) divided by the total number of words.
minlongpause = 0.400
# Source: Number of silent regions between words that are > 400 ms (Le et al., 2018; Pakhomov et al., 2010)
# divided by the total number of words.
concatenation_overlap_time = 0.02
# Overlap time (s) = the time by which any two adjacent sounds will come to overlap,
        # i.e. the time during which the earlier sound fades out and the later sound fades in.
# Source: https://www.fon.hum.uva.nl/praat/manual/Sounds__Concatenate_with_overlap___.html