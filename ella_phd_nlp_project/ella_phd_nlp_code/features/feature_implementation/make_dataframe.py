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

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_df_subject_task_features_updated(
    text_dir: str,
    audio_dir: str,
    tables_dir: str,
    excel_name="df_subject_task_features.xlsx",
    normalize = True,
) -> pd.DataFrame:
    """
    Build a dataframe with all subject-task pairs and features (z-normalized values).
    Rows: subject-task pairs
    Columns: features (z-scores: mean 0, std 1)

    :param text_dir: path to the processed text directory
    :param audio_dir: path to the processed audio directory
    :param tables_dir: path to the tables directory
    :param excel_name: filename for Excel output (default: df_subject_task_features.xlsx)
    :param normalize: whether to normalize the features (default: True)
    :return: a Pandas dataframe with normalized features, saved to Excel
    """

    ## Get the list of subject-task pairs from text_dir (or audio_dir if preferred)
    subject_task_pairs = get_subject_question_names_from_files(text_dir)
    index = pd.MultiIndex.from_tuples(subject_task_pairs, names=["subject_id", "task_id"])
    # these are subject-task pair indices: each index entry is a (subject_id, task_id) typle

    ## Initialize empty DataFrame with multi-index
    df_features = pd.DataFrame(index=index)

    ##  Helper function: Map features to dict keyed by (subject, task): this generates feature dictionaries for each feature
    def feature_to_dict(feature_func, *args, **kwargs):
        results = feature_func(*args, **kwargs)
        keys = get_subject_question_names_from_files(text_dir)  # or audio_dir depending on feature_func
        return dict(zip(keys, results))
        # The result is a dictionary for this function:{
        #   (subject1, task1): value1,
        #   (subject1, task2): value2,
        #   (subject2, task1): value3,
        #   ...
        # }

    ## Map features to columns in df_features: add each feature as a column by mapping from the dictionary
    #  1) .map() takes each index tuple and looks it up in the dictionary of the specific feature function (eg sem par).
    #  2) It returns the corresponding feature value.
    #  3) This creates a new column (eg "SEM_Semantic Paraphasias") with the feature values aligned to the correct
    #  (subject_id, task_id) rows.

    df_features["SEM_Semantic Paraphasias"] = df_features.index.map(
        feature_to_dict(semantic_paraphasias, text_dir)
    )
    print('semantic paraphasias calculated')
    df_features["PHON_Phonemic Paraphasias"] = df_features.index.map(
        feature_to_dict(phonemic_paraphasias, text_dir)
    )
    print('phonemic paraphasias calculated')
    df_features["PHON_Neologisms"] = df_features.index.map(
        feature_to_dict(neologisms, text_dir)
    )
    print('neologisms calculated')
    df_features["LEX_Number of Words"] = df_features.index.map(
        feature_to_dict(number_of_words, text_dir)
    )
    print('nb of words calculated')
    df_features["LEX_Brunet's Index"] = df_features.index.map(
        feature_to_dict(brunets_index, text_dir)
    )
    print('BI calculated')
    df_features["LEX_Noun Rate"] = df_features.index.map(
        feature_to_dict(noun_rate, text_dir)
    )
    print('noun rate calculated')
    df_features["LEX_Verb Rate"] = df_features.index.map(
        feature_to_dict(verb_rate, text_dir)
    )
    print('verb rate calculated')
    df_features["LEX_Adjective Rate"] = df_features.index.map(
        feature_to_dict(adjective_rate, text_dir)
    )
    print('adjective rate calculated')
    df_features["LEX_Pronoun Rate"] = df_features.index.map(
        feature_to_dict(pronoun_rate, text_dir)
    )
    print('pronoun rate calculated')
    df_features["LEX_Adverb Rate"] = df_features.index.map(
        feature_to_dict(adverb_rate, text_dir)
    )
    print('adverb rate calculated')
    df_features["LEX_Determiner Rate"] = df_features.index.map(
        feature_to_dict(determiner_rate, text_dir)
    )
    print('determiner rate calculated')
    df_features["LEX_Conjunction Rate"] = df_features.index.map(
        feature_to_dict(conjunction_rate, text_dir)
    )
    print('conjunction rate calculated')
    df_features["LEX_Preposition Rate"] = df_features.index.map(
        feature_to_dict(preposition_rate, text_dir)
    )
    print('preposition rate calculated')
    df_features["LEX_Particle Rate"] = df_features.index.map(
        feature_to_dict(particle_rate, text_dir)
    )
    print('particle rate calculated')
    df_features["LEX_Content-Function Ratio"] = df_features.index.map(
        feature_to_dict(content_function_ratio, text_dir)
    )
    print('content function ratio calculated')
    df_features["GRAM_MLU"] = df_features.index.map(
        feature_to_dict(mean_length_utterance, text_dir)
    )
    print('mean length utterance calculated')
    df_features["GRAM_Noun-Verb Rate"] = df_features.index.map(
        feature_to_dict(noun_verb_rate, text_dir)
    )
    print('noun verb rate calculated')
    df_features["GRAM_Subordinate Clauses"] = df_features.index.map(
        feature_to_dict(subordinate_clauses, text_dir)
    )
    print('subordinate clauses calculated')
    # Add # "GRAM_Syntactic Deviation" similarly..

    df_features["FLU_Filled Pause_T"] = df_features.index.map(
        feature_to_dict(filled_pauses, text_dir)
    )
    print('filled pauses calculated')
    df_features["FLU_False Start_T"] = df_features.index.map(
        feature_to_dict(false_starts, text_dir)
    )
    print('false starts calculated')
    df_features["FLU_Word Abandoned_T"] = df_features.index.map(
        feature_to_dict(word_abandoned, text_dir)
    )
    print('word abandoned calculated')
    df_features["FLU_Word Repetition_T"] = df_features.index.map(
        feature_to_dict(word_repetition, text_dir)
    )
    print('word repetition calculated')
    df_features["FLU_Speech Rate Words_AT"] = df_features.index.map(
        feature_to_dict(speech_rate_words, audio_dir, text_dir)
    )
    print('speech rate words AT calculated')
    df_features["FLU_Speech Rate Syllables_A"] = df_features.index.map(
        feature_to_dict(speech_rate_syllables, audio_dir)
    )
    print('speech rate syllables A calculated')
    df_features["FLU_Short Pauses_AT"] = df_features.index.map(
        feature_to_dict(silent_pauses, audio_dir, text_dir, short_or_long='short')
    )
    print('silent pauses calculated')
    df_features["FLU_Long Pauses_AT"] = df_features.index.map(
        feature_to_dict(silent_pauses, audio_dir, text_dir, short_or_long='long')
    )
    print('silent pauses calculated')
    df_features["FLU_Short Pause Rate_A"] = df_features.index.map(
        feature_to_dict(silent_pauses_rate,     audio_dir, short_or_long='short')
    )
    print('silent pause rate calculated')
    df_features["FLU_Long Pause Rate_A"] = df_features.index.map(
        feature_to_dict(silent_pauses_rate, audio_dir, short_or_long='long')
    )
    print('silent pause rate calculated')

    ## if normalize == True, Normalize all feature columns (ignore NaNs) to z-scores (mean = 0, std = 1)
    if normalize:
        scaler = StandardScaler()
        df_features[:] = scaler.fit_transform(df_features.values)
        # 1) df_features.values extracts the underlying numpy array of the dataframe values only (no index or column labels).
        # It’s a 2D array with shape (num_rows, num_features).
        # 2) scaler.fit_transform: i) fit (i) computes the mean and stdv for each column (feature), ii) transform() scales
        # the values using those mean and stdv statistics
        # 3) df_features[:] = ...
        # This assigns the normalized numpy array back in place to the original dataframe.
        # The [:] syntax means you replace all values in df_features without changing its structure (index and columns stay intact).
        # So the dataframe now holds the z-scored feature values.
        # Note: normally the df only has numeric values (feature outcomes), as the tuple-indices are not yet set to
        # separate columns (only happens later). It is thus safe to normalize the columns as it is only feature outcomes.
        base, ext = os.path.splitext(excel_name)  # ext includes the dot, e.g., ".xlsx"
        excel_name = f"{base}_normalized{ext}"
        # excel_name = '_'.join([excel_name.split('.')[0], "normalized", "xlsx"])

    ## Save to Excel (reset index to have subject_id and task_id as columns)
    output_path = os.path.join(tables_dir, excel_name)
    df_features.reset_index().to_excel(output_path, index=False)
    # Calling df_features.reset_index() moves the index levels back into regular columns.
    # So if your index is (subject_id, task_id), after reset_index() the dataframe will have columns: subject_id | task_id |...
    # Note: The index=False argument tells pandas not to write the dataframe index as an extra unnamed column.

    print(f"Saved feature dataframe to {output_path}")

    return df_features.reset_index()




def build_df_subject_task_features_OLD(
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
    file_name = os.path.join(TABLES_DIR, excel_name)
    df_features.to_excel(file_name, index=False)
    # https://www.geeksforgeeks.org/exporting-a-pandas-dataframe-to-an-excel-file/

    return df_features


def build_df_subject_task_features_absolute_OLD(
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


        "FLU_Filled Pause_T": filled_pauses(text_dir),
        "FLU_False Start_T": false_starts(text_dir),
        "FLU_Word Abandoned_T": word_abandoned(text_dir),
        "FLU_Word Repetition_T": word_repetition(text_dir),
        "FLU_Speech Rate Words_AT": speech_rate_words(audio_dir, text_dir),
        "FLU_Speech Rate Syllables_A": speech_rate_syllables(audio_dir),
        "FLU_Short Pauses_AT": silent_pauses(audio_dir, text_dir, 'short'),
        "FLU_Long Pauses_AT": silent_pauses(audio_dir, text_dir, 'long'),
        "FLU_Short Pause Rate_A": silent_pauses_rate(audio_dir, 'short'),
        "FLU_Long Pause Rate_A": silent_pauses_rate(audio_dir, 'long'),

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
    tables_dir: str,
    excel_name_df_subject_task_features: str = "df_subject_task_features.xlsx",
    excel_name: str = "df_subject_features_per_task.xlsx"
):
    """
    Generate a dataframe with all subjects and features, per task
    - sheets: per task, one separate sheet
    - rows: subjects
    - columns: features

    :param tables_dir: the tables directory
    :param excel_name_df_subject_task_features: the original excel file,
    defaults to df_subject_task_features.xlsx,
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
    output_path = os.path.join(tables_dir, excel_name)
    with pd.ExcelWriter(output_path) as writer:
        for task_name, task_df in task_dfs.items():
            task_df.to_excel(writer, sheet_name=str(task_name), index=False)

    return task_dfs



if __name__ == "__main__":
    text_dir = TEXT_DIR
    audio_dir = AUDIO_PATIENTU_DIR
    tables_dir = TABLES_DIR

    build_df_subject_task_features_updated(text_dir, audio_dir, tables_dir,
                                           excel_name = "df_subject_task_features.xlsx",
                                           normalize=True)
    build_df_subject_features_per_task(tables_dir,
                                        excel_name_df_subject_task_features = "df_subject_task_features_normalized.xlsx",
                                        excel_name= "df_subject_features_per_task_normalized.xlsx")

    build_df_subject_task_features_updated(text_dir, audio_dir, tables_dir,
                                           excel_name="df_subject_task_features.xlsx",
                                           normalize=False)

    build_df_subject_features_per_task(tables_dir,
                                       excel_name_df_subject_task_features="df_subject_task_features.xlsx",
                                       excel_name="df_subject_features_per_task.xlsx")



    # build_df_subject_task_features_updated(text_dir, audio_dir, tables_dir, normalize = False)
    """"
    build_df_subject_task_features(text_dir, audio_dir, tables_dir)
    build_df_subject_features_per_task(text_dir, audio_dir, tables_dir)

    build_df_subject_task_features_absolute(text_dir, audio_dir, tables_dir)
    build_df_subject_features_per_task(text_dir, audio_dir, tables_dir,
                                       excel_name_df_subject_task_features='df_subject_task_features_absolute.xlsx',
                                       excel_name='df_subject_features_per_task_absolute.xlsx'
                                       )
    """