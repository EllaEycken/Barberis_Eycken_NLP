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
TEXT_DIR = os.path.join(DATA_DIR, 'raw', 'transcripts')
TEXT_DIR_DUMMY = os.path.join(DATA_DIR, 'raw', 'transcripts_dummy')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

## Directory for models
MODELS_DIR = os.path.join(PROJECT_DIR, '../models')
MILTENBURG_MODEL_PATH = os.path.join(MODELS_DIR, 'miltenburg', 'model.perc.dutch_tagger_large.pickle')


## Directories for reports
REPORTS_DIR = os.path.join(PROJECT_DIR, '../reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
TABLES_DIR = os.path.join(REPORTS_DIR, 'tables')


## Q-file mappings
Q_FILE_MAPPINGS = {
    "q1_subject_wise": "",  # if looking for question 1, then looking at files with 'bio' in them etc
    "q2_subject_wise": "day",  # important: subject = participant!!
    "q3_subject_wise": "actua",
    "q4_subject_wise": "object",
    "q5_subject_wise": "picture",
}
