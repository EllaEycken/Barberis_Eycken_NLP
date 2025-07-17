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
from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_helper_functions.helper_extraction_audio import (
    calculate_duration,
    create_textGridDataframe,
    create_textGridSilencesObject,
)
from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import (
    read_sounds,
    read_transcripts,
)

""" FLUENCY """
def calculate_speech_rate(
  file_path: str,
):
    """Calculate the speech Rate (syll/s) using PRAAT via Parselmouth (based on Dr Feinberg).

    :param file_path: string
    :return: list of floats containing the Speech Rate aka Speaking Rate =
    number of syllables (that have been spoken)/ total duration of the signal
    NOTE: is not the articulation rate: Articulation Rate = number of syllables/ phonation or 'sounding' duration
    (without silences in the original sound)
    (definitions from sources here underneath)
    in the audio files, using PRAAT via Parselmouth (based on and adapted from Dr Feinberg)
    https://github.com/drfeinberg/PraatScripts/blob/master/syllable_nuclei.py
    = Praat Script Syllable Nuclei: Copyright (C) 2008  Nivja de Jong and Ton Wempe
    Based on paper:
    https://www.researchgate.net/publication/24274554_Praat_script_to_detect_syllable_nuclei_and_measure_speech_rate_automatically
    Large code: see helper_functions praat.py


    """
    speechRate_list, __, __, __ = calculate_syllNucleiFeinberg(file_path)
    # call the function to calculate the speaking rate, articulation rate, pause rate and avDurSyll:
    # specify which file directory (AUDIO_DIR so the same file directory as the file_path in this function
    # so file_path), specify whether to calculate speaking rate, articulation rate, pause rate or avDurSyll:
    # easiest is to 'ignore the 2nd, third and last return value (aka articulation rate, pause rate and avDurSyll)
    # How to ignore: by using __:
    # Function returns value a, b, c and d: a, b, c, d = function(...)
    # => only return a: a, __, __, __ = function(...)
    # from: https://stackoverflow.com/questions/431866/ignore-python-multiple-return-value
    # Result: speakingRate_list = a list of speaking Rates over all audio-files

    return speechRate_list
)



def calculate_propSilentPauses(
    file_path: str,
):
    """Calculate the proportion of silent Pauses using PRAAT via Parselmouth (inspired by Dr Feinberg).

    :param file_path: string
    :return: list of floats containing the percentage of silent pauses (s) compared to the total duration of the audio
    signal
    # TODO: pas aan van total duration naar total amount of words
    using PRAAT via Parselmouth (inspired by Dr Feinberg)
    https://github.com/drfeinberg/PraatScripts/blob/master/Measure%20Pitch%2C%20HNR%2C%20Jitter%2C%20Shimmer%2C%20and%20Formants.ipynb
    and based on the example of:
    https://stackoverflow.com/questions/34770105/praat-script-to-remove-silence-cannot-select-and-remove-objects
    and
    https://github.com/stylerw/styler_praat_scripts/blob/master/extract_silences.praat
    """
    propSilentPauses_list = []  # define a now still empty list of silent pauses across the audio-files
    list_of_sounds = read_sounds(file_path)  # make a list of sounds with the read-function
    for sound in list_of_sounds:  # for each item in this list of sounds
        silentPauses_list = []
        duration = calculate_duration(sound)  # calculate the duration of the sound
        textGridSilencesObject = create_textGridSilencesObject(sound)
        # make a textgrid object that distinguishes between sounding and silent intervals
        df_silences_af = create_textGridDataframe(textGridSilencesObject)
        # turn this textgrid into a pandas dataframe to make it readable in Python
        df_silentOnly = df_silences_af[df_silences_af["text"].str.contains("silent")]
        # keep only the 'silent' rows in the dataframe of this audio file
        # from: https: // saturncloud.io / blog / how - to - filter - pandas - dataframes - by - column - of - strings
        # /  #:~:text=Filtering%20by%20a%20Single%20String,string%20value%20in%20the%20column.
        for i in range(0, len(df_silentOnly)):
            tstart = df_silentOnly.iloc[i]["tmin"]
            tstop = df_silentOnly.iloc[i]["tmax"]
            # https://stackoverflow.com/questions/16729574/how-can-i-get-a-value-from-a-cell-of-a-dataframe
            deltat = tstop - tstart
            silentPauses_list.append(deltat)
        totalSilentPauses = sum(silentPauses_list)

        propSilentPauses = totalSilentPauses / duration

        propSilentPauses_list.append(propSilentPauses)
        # append this prop of silent pauses from this audio-file
        # to a list of prop of silent pauses across all audio-files

    return propSilentPauses_list




"""RUNNING THE FUNCTIONS"""
if __name__ == "__main__":
    text_dir = TEXT_DIR_DUMMY