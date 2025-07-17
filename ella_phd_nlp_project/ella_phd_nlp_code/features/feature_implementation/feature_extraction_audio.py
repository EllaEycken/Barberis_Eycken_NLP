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

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (  # TODO: if all is finished, switch this to AUDIO_PATIENT_DIR and TEXT_DIR!
    AUDIO_PATIENTU_DIR_DUMMY,
    TEXT_DIR_DUMMY,
    period_ceiling,
    unit,
    minpause,
    minshortpause, maxshortpause,
    minlongpause,

)
from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_helper_functions.helper_extraction_audio import (
    calculate_duration,
    create_textGridDataframe,
    create_textGridSilencesObject, calculate_syllNucleiFeinberg,
)
from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import (
    read_sounds,
    read_transcripts, get_subject_question_names_from_files,
)
from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_helper_functions.helper_extraction_text import (
    TokenCounter,
)


""" FLUENCY """
def calculate_speech_rate_words(
    audio_dir: str,
    transcript_dir: str,
):
    """Calculate the speech Rate (words/min) using PRAAT via Parselmouth

    :param audio_dir: the audio directory (containing the patientonly audio files)
    :param transcript_dir: the transcript directory (containing the transcripts)
    :return: list of floats containing the Speech Rate (word-level) =
    number of words (that have been spoken)/ total duration of the signal

    NOTE: is not the articulation rate: Articulation Rate = number of syllables (or words)/ phonation or 'sounding' duration
    (without silences in the original sound)
    (definitions from sources here underneath)
    in the audio files, using PRAAT via Parselmouth
    Based on paper:
    https://www.researchgate.net/publication/24274554_Praat_script_to_detect_syllable_nuclei_and_measure_speech_rate_automatically

    Large code: see helper_functions praat.py
    """
    speechRate_list = []  # define a now still empty list of speech rates across the audio-files

    # Define variables
    list_of_sounds = read_sounds(audio_dir)  # make a list of sounds with the read-function
    list_of_transcripts = read_transcripts(transcript_dir) # make a list of transcripts with the read-function
    list_of_subject_question_pairs_audio = get_subject_question_names_from_files(audio_dir)
    list_of_subject_question_pairs_transcripts = get_subject_question_names_from_files(transcript_dir)

    # Go over each sound
    for sound in list_of_sounds:  # for each item in this list of sounds
        # Find the subject and question corresponding to that sound
        sound_index = list_of_sounds.index(sound)
        (subject_name_audio, question_name_audio) = list_of_subject_question_pairs_audio[sound_index]

        # Find the transcript corresponding to that subject and question
        transcript_index = None
        transcript = None
        for (subject_name_transcript, question_name_transcript) in list_of_subject_question_pairs_transcripts:
            if subject_name_audio == subject_name_transcript:
                if question_name_audio == question_name_transcript:
                    transcript_index = list_of_subject_question_pairs_transcripts.index([subject_name_transcript,
                                                                                         question_name_transcript])
                    transcript = list_of_transcripts[transcript_index]
                    break


        # Calculate the speech rate
        total_nb_of_words = TokenCounter(transcript).total_number_of_words()
        total_duration = calculate_duration(sound)
        speechRateSec = total_nb_of_words / total_duration
        speechRateMin = speechRateSec * 60
        speechRate_list.append(speechRateMin)

    return speechRate_list




def calculate_speech_rate_syllables(
  audio_dir: str,
):
    """Calculate the speech Rate (syll/min) using PRAAT via Parselmouth (based on Dr Feinberg).

    :param audio_dir: the audio directory (containing the patientonly audio files)
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
    speechRateMin_list = []
    speechRateSec_list, __, __, __ = calculate_syllNucleiFeinberg(audio_dir)
    # call the function to calculate the speaking rate, articulation rate, pause rate and avDurSyll:
    # specify which file directory (AUDIO_DIR so the same file directory as the file_path in this function
    # so file_path), specify whether to calculate speaking rate, articulation rate, pause rate or avDurSyll:
    # easiest is to 'ignore the 2nd, third and last return value (aka articulation rate, pause rate and avDurSyll)
    # How to ignore: by using __:
    # Function returns value a, b, c and d: a, b, c, d = function(...)
    # => only return a: a, __, __, __ = function(...)
    # from: https://stackoverflow.com/questions/431866/ignore-python-multiple-return-value
    # Result: speakingRate_list = a list of speaking Rates over all audio-files


    for item in speechRateSec_list:
        itemMin = item * 60
        speechRateMin_list.append(itemMin)

    return speechRateMin_list


def calculate_SilentPauseProportion(
    audio_dir: str,
    transcript_dir: str,
    short_or_long: str,
):
    """Calculate the proportion of SHORT silent Pauses (#/word) using PRAAT via Parselmouth (inspired by Dr Feinberg).

    :param audio_dir: the audio directory (containing the patientonly audio files)
    :param transcript_dir: the transcript directory (containing the transcripts)
    :param short_or_long: the short or long silence
    :return: list of floats containing the proportion of SHORT silent pauses (#/word)

    For SHORT pauses: Based on the following definition:
    Number of silent regions between words that are 150-400 ms (Le et al., 2018; Pakhomov et al., 2010)
    divided by the total number of words.

    For LONG pauses: Based on the following definition:
    Number of silent regions between words that are > 400 ms (Le et al., 2018; Pakhomov et al., 2010)
    divided by the total number of words.

    using PRAAT via Parselmouth (inspired by Dr Feinberg)
    https://github.com/drfeinberg/PraatScripts/blob/master/Measure%20Pitch%2C%20HNR%2C%20Jitter%2C%20Shimmer%2C%20and%20Formants.ipynb
    and based on the example of:
    https://stackoverflow.com/questions/34770105/praat-script-to-remove-silence-cannot-select-and-remove-objects
    and
    https://github.com/stylerw/styler_praat_scripts/blob/master/extract_silences.praat
    """
    propSilentPauses_list = []  # define a now still empty list of proportions of silent pauses across the audio-files

    # Define the variables
    list_of_sounds = read_sounds(audio_dir)  # make a list of sounds with the read-function
    list_of_transcripts = read_transcripts(transcript_dir)  # make a list of transcripts with the read-function
    list_of_subject_question_pairs_audio = get_subject_question_names_from_files(audio_dir)
    list_of_subject_question_pairs_transcripts = get_subject_question_names_from_files(transcript_dir)

    # Go over each sound
    for sound in list_of_sounds:  # for each item in this list of sounds
        ## Find the subject and question corresponding to that sound
        sound_index = list_of_sounds.index(sound)
        (subject_name_audio, question_name_audio) = list_of_subject_question_pairs_audio[sound_index]

        ## Find the transcript corresponding to that subject and question and compute the total number of words
        transcript_index = None
        transcript = None
        for (subject_name_transcript, question_name_transcript) in list_of_subject_question_pairs_transcripts:
            if subject_name_audio == subject_name_transcript:
                if question_name_audio == question_name_transcript:
                    transcript_index = list_of_subject_question_pairs_transcripts.index([subject_name_transcript,
                                                                                         question_name_transcript])
                    transcript = list_of_transcripts[transcript_index]
                    break
        total_nb_of_words = TokenCounter(transcript).total_number_of_words()


        ## Calculate the proportion of silent pauses for this sound

        # Filter the silent parts of the audio signal into one 'silence' dataframe
        textGridSilencesObject = create_textGridSilencesObject(sound) # make a textgrid object that distinguishes between sounding and silent intervals
        df_silences_af = create_textGridDataframe(textGridSilencesObject) # turn this textgrid into a pandas dataframe to make it readable in Python
        df_silentOnly = df_silences_af[df_silences_af["text"].str.contains("silent")] # keep only the 'silent' rows in the dataframe of this audio file
        # from: https: // saturncloud.io / blog / how - to - filter - pandas - dataframes - by - column - of - strings
        # /  #:~:text=Filtering%20by%20a%20Single%20String,string%20value%20in%20the%20column.

        # Extract the  silent pauses from each silent segment (row) in the dataframe
        silentPauses_list = []
        for i in range(0, len(df_silentOnly)): # go over the rows of the df
            tstart = df_silentOnly.iloc[i]["tmin"]  # note: is in SECONDS
            tstop = df_silentOnly.iloc[i]["tmax"]
            # https://stackoverflow.com/questions/16729574/how-can-i-get-a-value-from-a-cell-of-a-dataframe
            deltat = tstop - tstart
            if short_or_long == "short":
                if minshortpause<= deltat <= maxshortpause:  # note: is all in SECONDS
                    silentPauses_list.append(deltat)
                else:
                    continue
            else: # so looking at long pauses
                if minlongpause < deltat:  # note: is all in SECONDS
                    silentPauses_list.append(deltat)
                else:
                    continue


        # Calculate the proportion of silent pauses in for this patient-task file
        total_amount_silent_pauses = len(silentPauses_list)
        prop_silent_pauses = total_amount_silent_pauses/ total_nb_of_words
        propSilentPauses_list.append(prop_silent_pauses)
        # append this prop of silent pauses from this audio-file
        # to a list of prop of silent pauses across all audio-files


    return propSilentPauses_list



def calculate_SilentPausesRate(
    audio_dir: str,
):
    """Calculate the proportion of silent Pauses using PRAAT via Parselmouth (inspired by Dr Feinberg).

    :param audio_dir: the audio directory (containing the patientonly audio files)
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
    list_of_sounds = read_sounds(audio_dir)  # make a list of sounds with the read-function
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
    audio_dir = AUDIO_PATIENTU_DIR_DUMMY
    transcript_dir = TEXT_DIR_DUMMY

    # calculate_speech_rate_words(audio_dir, transcript_dir)
    # calculate_speech_rate_syllables(audio_dir)
    # calculate_SilentPauseProportion(audio_dir, transcript_dir, 'short')
    calculate_SilentPauseProportion(audio_dir, transcript_dir, 'long')