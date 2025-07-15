""" Preprocess audio files from IANSA dataset """

import os
import re
import glob


import datasets  # install it first: is a package

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    AUDIO_DIR_DUMMY, DIAR_DIR_DUMMY, CLEAN_DIAR_DIR_DUMMY)  # TODO: swap for non-dummy directories once in order

from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import *
from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_helper_functions.helper_extraction_audio import *

## Create a helper function that checks diarization_files
def cleanup_diar_txt_file(diar_txt_path_in, interim_dir):
    """
    Helper function to clean up the diar txt file
    :param diar_txt_path_in: the diarization.txt path (showing the indices per speaker: start, end, spk) that must
    be cleaned up
    :param interim_dir: the interim directory path where the diar_txt file must be temporarily saved
    :return: the text file in the interim directory, subject to the following changes
            1) only keep spk 0 and 1
    """
    ## Choose the name for the cleaned diar_text file
    txt_name_subparts = os.path.splitext(os.path.basename(diar_txt_path_in))[0].split('_')
    # 'os.path.splitext(os.path.basename(diar_txt_path_in))[0]' extracts the name of the diar_txt_path
    # (all without the .txt extension)
    diar_txt_correct_name = '_'.join(txt_name_subparts)
    # make sure you only keep the 'sub-XXX', 'transcriptie' and 'NAME TASK' (so discard the last 'preclean' part) and
    # join them
    diar_txt_path_out = os.path.join(
        interim_dir, ".".join(
            ["_".join([str(diar_txt_correct_name), 'diarization']),
             'txt']))

    with (open(diar_txt_path_in, 'r') as infile, open(diar_txt_path_out, 'w') as outfile):
        # 'r' = open the txt_path_in file for reading, 'w' = open the txt_path_out for writing
        # Note: encoding as utf-8 is needed as Python= , encoding = 'utf-8'
        lines = infile.readlines()
        if not lines:
            raise ValueError("The input file is empty")

        for line in lines:
            # Remove lines ending with nbr 2, because we cannot have more than 2 speakers
            if line.endswith('2'):
                continue

            # Write the modified line to the output file, IF the line is not empty
            if line:
                outfile.write(line)

    return diar_txt_path_out


## Create a helper function that returns the binary speaker code of the patient
def give_patient_spk_code(diar_txt_path_in_clean):
    """
    Helper function to return the binary speaker code of the patient
    :param diar_txt_path_in_clean: the CLEAN diarization.txt path (showing the indices per speaker: start, end, spk)
    :return: the binary speaker code of the patient (0 or 1)
    """
    first_line_diar_code = int()
    spk_code = int()
    with open(diar_txt_path_in_clean, 'r') as f:
        first_line = f.readline().strip()  # Read only the first line and strip whitespace

        # line format is: "start,end,speaker" -> we split by ','
        parts = first_line.split(',')

        # Get the last element, which should be the speaker code
        first_line_diar_code = int(parts[-1])

    if first_line_diar_code == 0:  # we assume first line will be the test administrator, so the first spk is NOT who we want
        spk_code = int(1)  # if the first speaker (test admin) has spk-code 0, then the patient has code 1...
    if first_line_diar_code == 1:
        spk_code = int(0)  #... and vice versa

    return int(spk_code)



## Filter audio file based on the diarization indices
def filter_audio_file(raw_audio_file, diarization_dir, processed_dir, overrule_spk_code = False):
    """

    :param raw_audio_file: the original audio (wav) file that is stored in the raw directory
    :param diarization_dir: the diarization directory path with diarization files
    :param processed_dir: the processed directory where the filtered audio files will be saved
    :param overrule_spk_code: whether to overwrite spk code if it exists
    :return: a filtered audio file in the processed directory, containing only audio intervals (concatenated) attributed
    to the patient
    """






## Preprocess all IANSA audio files (by filtering them)
def preprocess_IANSA_audio(raw_dir, interim_dir, processed_dir):
    """
    Preprocessing function to filter the audio files based on the speaker diarization codes
    :param audio_dir: the raw audio directory containing the audio (wav) files
    :param interim_dir: the interim directory path where the (clean) diar_txt file are saved
    :param processed_dir: the processed directory path where the filtered audio files will be stored
    :return: the audio file in the processed directory, containing only the sounds of the speaker
    """


if __name__ == "__main__":

    diar_txt_path_in = os.path.join(DIAR_DIR_DUMMY,'sub-a043_ANTAT.txt')
    interim_dir = CLEAN_DIAR_DIR_DUMMY
    cleanup_diar_txt_file(diar_txt_path_in, interim_dir)
    give_patient_spk_code(diar_txt_path_in)
    """
    docx_path = os.path.join(DOCX_DIR_DUMMY,'sub-a043_transcriptie_MCA.docx')
    interim_dir = PRECLEANTEXT_DIR_DUMMY
    convert_docx_to_txt(docx_path, interim_dir)
    txt_path_in = os.path.join(interim_dir, 'sub-a043_transcriptie_MCA_preclean.txt')
    processed_dir = TEXT_DIR_DUMMY
    cleanup_txt_file(txt_path_in, processed_dir)
    """


