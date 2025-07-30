"""Make a dataframe to list all extracted features for the participants and the tasks.

The dataframes will have the following structure:
- rows = subjects-tasks of the study
- columns = extracted features
"""

""" IMPORTS """
import os
import epitran
import pandas as pd
import spacy

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    AUDIO_PATIENTU_DIR, AUDIO_PATIENTU_DIR_DUMMY,
    TEXT_DIR, TEXT_DIR_DUMMY,
)

from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_implementation.feature_extraction_text import *
from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_implementation.feature_extraction_audio import *


from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import (
    get_subject_question_names_from_files,
)


""" SET UP NLP"""
## Call NLP
# epi = epitran.Epitran("nld-Latn")
nlp = spacy.load("nl_core_news_lg")
# Apply my custom fixer
fixer = FixNlpPipeline(nlp)
fixer.fix_nlp_pipeline()


""" HELPER FUNCTIONS """

def color_if_even(
    s
):
    """Color the rows of the dataframe if the row number is even.

    :param s: series of the dataframe
    """
    return ["background-color: grey" if i != i - 1 else "background-color: white" for i in s]
    # is useful later to give all rows from 1 subject (s08) for example one
    # specific color


# from https://stackoverflow.com/questions/39824472/using-pandas-style-to-give-colors-to-some-rows-with-a-specific
# -condition

def normalize_results_z(input_list: list):
    """Calculate the Z-values (m=0, std =1) of the input values by means of Z-scaling.

    :param input_list: string
    :return: list of floats containing the (normalized) Z-values of the original values,
    with zero mean and unit variance, using the following formula:
    X_norm or Z_value = (X - X.mean())/X.std()
    from: https://pieriantraining.com/tutorial-how-to-normalize-data-in-python/
    Source: my repo masterthesisEllaLAW

    """
    z_value_list = []
    x_mean = statistics.mean(input_list)
    x_std = statistics.stdev(input_list)

    for x in input_list:
        if x_std == 0:  # so there is no deviation from the mean/average value
            z_value = 0  # no deviation means: z = 0
            z_value_list.append(z_value)
        else:
            z_value = (x - x_mean) / x_std
            z_value_list.append(z_value)

    return z_value_list



""" BUILD DATAFRAME """


def build_df_subject_task_features(
    text_dir: str,
    audio_dir: str,
) -> pd.DataFrame:
    """Build a dataframe with all subject-task-pairs and features.

    :param text_dir: path to the (processed) text directory (already in function)
    :type audio_dir: path to the (processed) audio directory
    :return: a dataframe covering ALL features (as z-values), with subjects-tasks pairs as rows,
    features as columns
    :rtype: pd.DataFrame
    """
    # ---- STEP 1: initialize lists of data ----
    data = {
        """ ROWS """
        "subject_id": [name[0] for name in get_subject_question_names_from_files(text_dir)],
        # use list comprehension ([...]) and take the first part of the 'subject and question name'
        # aka take the 'subject name'
        "question_id": [name[1] for name in get_subject_question_names_from_files(text_dir)],
        # use list comprehension ([...]) and take the second part of the 'subject and question name'
        # aka take the 'question name'

        """ COLUMNS"""
        "SEM_Semantic Paraphasias": normalize_results_z(semantic_paraphasias(text_dir)),

        "PHON_Phonemic Paraphasias": normalize_results_z(phonemic_paraphasias(text_dir)),
        "PHON_Neologisms": normalize_results_z(neologisms(text_dir)),

        "LEX_Number of Words": normalize_results_z(number_of_words(text_dir)),
        "LEX_Brunet's Index": normalize_results_z(brunets_index(text_dir)),
        "LEX_Noun Rate": normalize_results_z(noun_rate(text_dir)),
        "LEX_Verb Rate": normalize_results_z(verb_rate(text_dir)),
        "LEX_Adjective Rate": normalize_results_z(adjective_rate(text_dir)),
        "LEX_Pronoun Rate": normalize_results_z(pronoun_rate(text_dir)),
        "LEX_Adverb Rate": normalize_results_z(adverb_rate(text_dir)),
        "LEX_Determiner Rate": normalize_results_z(determiner_rate(text_dir)),
        "LEX_Conjunction Rate": normalize_results_z(conjunction_rate(text_dir)),
        "LEX_Preposition Rate": normalize_results_z(preposition_rate(text_dir)),
        "LEX_Particle Rate": normalize_results_z(particle_rate(text_dir)),
        "LEX_Content-Function Ratio": normalize_results_z(content_function_ratio(text_dir)),

        "GRAM_MLU": normalize_results_z(mean_length_utterance(text_dir)),
        "GRAM_Noun-Verb Rate": normalize_results_z(noun_verb_rate(text_dir)),
        "GRAM_Subordinate Clauses": normalize_results_z(subordinate_clauses(text_dir)),
        # "GRAM_Syntactic Deviation": normalize_results_Z(calculate_propof2CharLength(file_path)),

        "FLU_Filled Pause_T": normalize_results_z(filled_pauses(text_dir)),
        "FLU_False Start_T": normalize_results_z(false_starts(text_dir)),
        "FLU_Word Abandoned_T": normalize_results_z(word_abandoned(text_dir)),
        "FLU_Word Repetition_T": normalize_results_z(word_repetition(text_dir)),
        "FLU_Speech Rate Words_AT": normalize_results_z(speech_rate_words(audio_dir, text_dir)),
        "FLU_Speech Rate Syllables_A": normalize_results_z(speech_rate_syllables(audio_dir)),
        "FLU_Short Pauses_AT": normalize_results_z(silent_pauses(audio_dir, text_dir, 'short')),
        "FLU_Long Pauses_AT": normalize_results_z(silent_pauses(audio_dir, text_dir, 'long')),

    }

    # ---- STEP 2: Create Dataframe ----
    df_features = pd.DataFrame(data)
    # https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
    pd.set_option("display.max.columns", None)
    df_features.style.background_gradient().set_caption("Table of Features").apply(
        color_if_even, subset=["subject_id"]
    )

    # ---- STEP 3: save Dataframe as excel in interim data directory
    file_name = os.path.join(INTERIM_DIR, "df_nltk_lexicon.xlsx")
    df_lexical.to_excel(file_name, index=False)
    # https://www.geeksforgeeks.org/exporting-a-pandas-dataframe-to-an-excel-file/

    return df_lexical


def build_extra_df_for_character_counts(
    file_path: str,
) -> pd.DataFrame:
    """Generate a dataframe with only the character counts in the same style as the other function.

    :param file_path: text directory (already in function)
    :type file_path: str
    :return: a dataframe covering character counts, with participants-questions as rows,
    features as columns
    """
    # ---- STEP 1: initialize data of lists ----
    data_lexical = {
        "subject_id": [name[0] for name in get_subject_question_names_from_files(file_path)],
        # use list comprehension ([...]) and take the first part of the 'subject and question name'
        # aka take the 'subject name'
        "question_id": [name[1] for name in get_subject_question_names_from_files(file_path)],
        # use list comprehension ([...]) and take the second part of the 'subject and question name'
        # aka take the 'question name'
        "character_count": calculate_character_count(file_path),
    }

    # ---- STEP 2: Create Dataframe ----
    df_lexical = pd.DataFrame(data_lexical)
    # https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
    pd.set_option("display.max.columns", None)
    df_lexical.style.background_gradient().set_caption("Table of Language-Lexical Features").apply(
        color_if_even, subset=["subject_id"]
    )

    # ---- STEP 3: save Dataframe as excel in interim data directory
    file_name = os.path.join(INTERIM_DIR, "df_nltk_character.xlsx")
    df_lexical.to_excel(file_name, index=False)
    # https://www.geeksforgeeks.org/exporting-a-pandas-dataframe-to-an-excel-file/

    return df_lexical


if __name__ == "__main__":
    build_extra_df_for_character_counts(TEXT_DIR)