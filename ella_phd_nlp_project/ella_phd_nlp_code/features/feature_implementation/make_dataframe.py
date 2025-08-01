"""Make a dataframe to list all extracted features for the participants and the tasks.

The first dataframe will have the following structure:
- rows = subjects-tasks of the study
- columns = extracted features

The second dataframe will have the following structure:
- sheet: task
- rows = subjects of the study
- columns = extracted features
"""


""" IMPORTS """
import os
import epitran
import pandas as pd
import spacy
# Note: don't forget to install openpyxl package (pip install openpyxl in your anaconda prompt)


from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    AUDIO_PATIENTU_DIR, # AUDIO_PATIENTU_DIR_DUMMY,
    TEXT_DIR, # TEXT_DIR_DUMMY,
    TABLES_DIR
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
    tables_dir: str,
    excel_name = str("df_subject_task_features.xlsx"),
) -> pd.DataFrame:
    """Build a dataframe with all subject-task-pairs and features (z-normalized values).
    - rows: subject-task pairs
    - columns: features (z-scores: mean 0, std 1)

    :param text_dir: path to the (processed) text directory (already in function)
    :param audio_dir: path to the (processed) audio directory
    :param tables_dir: path to the tables directory
    :param excel_name: defaults to df_subject_task_features.xlsx
    :return: a Pandas dataframe covering ALL features (as z-values), with subjects-tasks pairs as rows,
    features as columns and an Excel file saved in the tables directory
    """
    # ---- STEP 1: initialize lists of data ----
    data = {
        "subject_id": [name[0] for name in get_subject_question_names_from_files(text_dir)],
        # use list comprehension ([...]) and take the first part of the 'subject and question name'
        # aka take the 'subject name'
        "task_id": [name[1] for name in get_subject_question_names_from_files(text_dir)],
        # use list comprehension ([...]) and take the second part of the 'subject and question name'
        # aka take the 'question name'

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


        # "FLU_Filled Pause_T": normalize_results_z(filled_pauses(text_dir)),
        # "FLU_False Start_T": normalize_results_z(false_starts(text_dir)),
        # "FLU_Word Abandoned_T": normalize_results_z(word_abandoned(text_dir)),
        # "FLU_Word Repetition_T": normalize_results_z(word_repetition(text_dir)),
        # "FLU_Speech Rate Words_AT": normalize_results_z(speech_rate_words(audio_dir, text_dir)),
        # "FLU_Speech Rate Syllables_A": normalize_results_z(speech_rate_syllables(audio_dir)),
        # "FLU_Short Pauses_AT": normalize_results_z(silent_pauses(audio_dir, text_dir, 'short')),
        # "FLU_Long Pauses_AT": normalize_results_z(silent_pauses(audio_dir, text_dir, 'long')),

    }

    # ---- STEP 2: Create Dataframe ----
    df_features = pd.DataFrame(data)
    # https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
    pd.set_option("display.max.columns", None)
    df_features.style.background_gradient().set_caption("Table of Features").apply(
        color_if_even, subset=["subject_id"]
    )

    # ---- STEP 3: save Dataframe as excel in interim data directory
    file_name = os.path.join(TABLES_DIR, excel_name)
    df_features.to_excel(file_name, index=False)
    # https://www.geeksforgeeks.org/exporting-a-pandas-dataframe-to-an-excel-file/

    return df_features


def build_df_subject_task_features_absolute(
    text_dir: str,
    audio_dir: str,
    tables_dir: str,
    excel_name = str("df_subject_task_features_absolute.xlsx"),
) -> pd.DataFrame:
    """Build a dataframe with all subject-task-pairs and features (absolute values).
    - rows: subject-task pairs
    - columns: features

    :param text_dir: path to the (processed) text directory (already in function)
    :param audio_dir: path to the (processed) audio directory
    :param tables_dir: path to the tables directory
    :param excel_name: defaults to df_subject_task_features.xlsx
    :return: a Pandas dataframe covering ALL features (as z-values), with subjects-tasks pairs as rows,
    features as columns and an Excel file saved in the tables directory
    """
    # ---- STEP 1: initialize lists of data ----
    data = {
        "subject_id": [name[0] for name in get_subject_question_names_from_files(text_dir)],
        # use list comprehension ([...]) and take the first part of the 'subject and question name'
        # aka take the 'subject name'
        "task_id": [name[1] for name in get_subject_question_names_from_files(text_dir)],
        # use list comprehension ([...]) and take the second part of the 'subject and question name'
        # aka take the 'question name'

        "SEM_Semantic Paraphasias": semantic_paraphasias(text_dir),

        "PHON_Phonemic Paraphasias": phonemic_paraphasias(text_dir),
        "PHON_Neologisms": neologisms(text_dir),

        "LEX_Number of Words": number_of_words(text_dir),
        "LEX_Brunet's Index": brunets_index(text_dir),
        "LEX_Noun Rate": noun_rate(text_dir),
        "LEX_Verb Rate": verb_rate(text_dir),
        "LEX_Adjective Rate": adjective_rate(text_dir),
        "LEX_Pronoun Rate": pronoun_rate(text_dir),
        "LEX_Adverb Rate": adverb_rate(text_dir),
        "LEX_Determiner Rate": determiner_rate(text_dir),
        "LEX_Conjunction Rate": conjunction_rate(text_dir),
        "LEX_Preposition Rate": preposition_rate(text_dir),
        "LEX_Particle Rate": particle_rate(text_dir),
        "LEX_Content-Function Ratio": content_function_ratio(text_dir),

        "GRAM_MLU": mean_length_utterance(text_dir),
        "GRAM_Noun-Verb Rate": noun_verb_rate(text_dir),
        "GRAM_Subordinate Clauses": subordinate_clauses(text_dir),
        # "GRAM_Syntactic Deviation": calculate_propof2CharLength(file_path),


        # "FLU_Filled Pause_T": filled_pauses(text_dir),
        # "FLU_False Start_T": false_starts(text_dir),
        # "FLU_Word Abandoned_T": word_abandoned(text_dir),
        # "FLU_Word Repetition_T": word_repetition(text_dir),
        # "FLU_Speech Rate Words_AT": speech_rate_words(audio_dir, text_dir),
        # "FLU_Speech Rate Syllables_A": speech_rate_syllables(audio_dir),
        # "FLU_Short Pauses_AT": silent_pauses(audio_dir, text_dir, 'short'),
        # "FLU_Long Pauses_AT": silent_pauses(audio_dir, text_dir, 'long'),

    }

    # ---- STEP 2: Create Dataframe ----
    df_features = pd.DataFrame(data)
    # https://stackoverflow.com/questions/18837262/convert-python-dict-into-a-dataframe
    pd.set_option("display.max.columns", None)
    df_features.style.background_gradient().set_caption("Table of Features").apply(
        color_if_even, subset=["subject_id"]
    )

    # ---- STEP 3: save Dataframe as excel in interim data directory
    file_name = os.path.join(TABLES_DIR, excel_name)
    df_features.to_excel(file_name, index=False)
    # https://www.geeksforgeeks.org/exporting-a-pandas-dataframe-to-an-excel-file/

    return df_features


def build_df_subject_features_per_task(
    text_dir: str,
    audio_dir: str,
    tables_dir: str,
    excel_name_df_subject_task_features = str("df_subject_task_features.xlsx"),
    excel_name = str("df_subject_features_per_task.xlsx")
):
    """
    Generate a dataframe with all subjects and features, per task
    - sheets: per task, one separate sheet
    - rows: subjects
    - columns: features

    :param text_dir: the (processed) text directory
    :param audio_dir: the (processed) audio directory
    :param tables_dir: the tables directory
    :param excel_name: defaults to df_subject_features_per_task.xlsx
    :return: a dictionary containing key-value pairs,
     - with key = task_name
     - with value = a Pandas dataframe of that task covering ALL features (as z-values), with subjects as rows,
        features as columns
    and an Excel file saved in the tables directory
    """
    df = pd.read_excel(os.path.join(tables_dir, excel_name_df_subject_task_features))
    # df= build_df_subject_task_features(text_dir, audio_dir, tables_dir))

    # Get the different tasks
    task_names = df['task_id'].unique()

    # Create a dictionary to hold each task-specific DataFrame
    task_dfs = {}

    for task_name in task_names:
        # Filter for this task
        df_task = df[df['task_id'] == task_name].copy()
        # Drop 'task' column
        df_task = df_task.drop(columns=['task_id'])
        # Store with task name as key
        task_dfs[task_name] = df_task

    # Write to Excel with multiple sheets
    file_name = os.path.join(TABLES_DIR, excel_name)
    with pd.ExcelWriter(file_name) as writer:
        for task_name, task_df in task_dfs.items():
            task_df.to_excel(writer, sheet_name=str(task_name), index=False)

    return task_dfs



if __name__ == "__main__":
    text_dir = TEXT_DIR
    audio_dir = AUDIO_PATIENTU_DIR
    tables_dir = TABLES_DIR

    # build_df_subject_task_features(text_dir, audio_dir, tables_dir)
    # build_df_subject_features_per_task(text_dir, audio_dir, tables_dir)
    build_df_subject_task_features_absolute(text_dir, audio_dir, tables_dir)

    build_df_subject_features_per_task(text_dir, audio_dir, tables_dir,
                                       excel_name_df_subject_task_features='df_subject_task_features_absolute.xlsx',
                                       excel_name='df_subject_features_per_task_absolute.xlsx'
                                       )
