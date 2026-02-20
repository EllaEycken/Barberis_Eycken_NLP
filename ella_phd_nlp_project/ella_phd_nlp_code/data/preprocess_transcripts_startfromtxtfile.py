""" Preprocess transcripts from IANSA dataset (Starting from TEXT FILES) """
""" Use this preprocessing file if the transcripts are TXT-files and not docx
    Mostly used if you want to use non-annotated transcripts. 
    Note that annotations are NOT fixed because it assumes there were none. """

import os
import re
import glob
from dataclasses import dataclass
from docx import Document  # first install python-docx
from typing import List

import datasets  # install it first: is a package

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    PRECLEANTEXT_DIR, TEXT_DIR)
    # DOCX_DIR_DUMMY, PRECLEANTEXT_DIR_DUMMY, TEXT_DIR_DUMMY,


def cleanup_txt_file(txt_path_in, processed_dir):
    """
    Helper function to clean up a text file
    :param txt_path_in: the text file path that must be cleaned up
    :param processed_dir: the processed directory path where the cleaned text files will be saved
    :return: returns the text file with the following changes
        1) Remove the title (first line) for
            ANTAT 1
            MCA transcriptie
        2) Remove the following lines: that start with.. 'ANTAT I', 'MCA transcriptie', 'Set', 'Item', 'Oefenitem'
                                                          'A:', 'Weekend', 'Stroke'
        3) Remove the 'A:' lines (= test leader)
        4) Remove the 'B:' parts in the participant lines
        5) Remove ggg, <spk, and xxx
        6) Remove [] in [...] statements unless it is an [A:…] statement, then remove it completely
        7) Remove dialect words, only keep normalization --> remove ( ) from normalization so that normalization is
        counted
        8) Remove direct speech annotations: :"..."
        9) Clean text formatting: Replace multiple spaces with a single space, Replace double punctuation with single,
         Remove  whitespace in between paragraphs, Add whitespace after '.' if necessary

    """
    ## Choose the name for the cleaned text file
    txt_name_subparts = os.path.splitext(os.path.basename(txt_path_in))[0].split('_')
    # 'os.path.splitext(os.path.basename(txt_path_in))[0]' extracts the name of the txt_path (all without the .txt
    # extension)
    txt_correct_name = str
    if txt_name_subparts[-1] != 'transcriptie':
        txt_correct_name = '_'.join(txt_name_subparts)
    if txt_name_subparts[-1] == 'transcriptie':
        txt_name_subparts[-2], txt_name_subparts[-1] = txt_name_subparts[-1], txt_name_subparts[-2]
        txt_correct_name = '_'.join(txt_name_subparts)

    txt_correct_name = '_'.join(txt_name_subparts[:-1])
    # make sure you only keep the 'sub-XXX', 'transcriptie' and 'NAME TASK' (so discard the last 'preclean' part) and
    # join them

    txt_path_out = os.path.join(
        processed_dir, ".".join(
            [txt_correct_name, 'txt']))


    with (
        open(txt_path_in, 'r', encoding = "utf-8") as infile,
        open(txt_path_out, 'w') as outfile
    ):
        # 'r' = open the txt_path_in file for reading, 'w' = open the txt_path_out for writing
        # Note: encoding as utf-8 is needed as Python should be able to read different characters , encoding = 'utf-8'

        # first_line = infile.readline()
        # if not first_line:
        #    raise ValueError("The input file is empty")

        lines = infile.readlines()
        if not lines:
            raise ValueError("The input file is empty")

        for line in lines:
            ## Remove abundant text
            line = line.replace('ggg', '').replace('<spk>', '').replace('xxx', ''
                                                                        ).replace(':"',''
                                                                                  ).replace('"','')
            # Remove coughs/laughs (transcribed as 'ggg') from transcript line
            # Remove <spk> transcription for speaker changes from transcript line
            # Remove direct speech annotations (:"")
            # replace 'xxx' words

            ## Remove 'allee'
            line = line.replace('allee','')

            ## Clean text formatting
            line = re.sub(r'\s+', ' ', line)  # Replace multiple spaces with a single space
            line = line.replace('. .', '.')  # Replace double punctuation with single
            line = line.strip()  # Strip leading/trailing whitespace
            line = re.sub(r'\.(?!\s)', '. ', line)  # Add whitespace after '.' if necessary


            # Write the modified line to the output file, IF the line is not empty
            if line:
                outfile.write(line)

    return txt_path_out


def preprocess_IANSA_transcripts_startfromtxtfile(interim_dir, processed_dir):
    """
    Preprocess ALL IANSA transcripts (using the helper function cleanup_txt_file)
    :param interim_dir: the interim directory where all the ***ORIGINAL*** TEXT files were stored
    :param processed_dir: the processed directory path where the cleaned TEXT files will be saved
    :return: a list of all the CLEANED text files in the interim directory
    """
    all_txt_files_list = list()
    all_txt_files_list = (
        glob.glob(interim_dir + f"{os.path.sep}sub-a[0-9]*")  # txt files from aphasia patients
        #  the 'sub-a*' looks for all files starting with sub-a
        #  [0-9] means 'any digit and doesn't matter how many digits;
        #  * means 'doesn't matter what comes after this'
        + glob.glob(interim_dir + f"{os.path.sep}sub-b[0-9]*") # txt files from control patients (comm partner)
        + glob.glob(interim_dir + f"{os.path.sep}sub-c[0-9]*")  # txt files from control patients (non-related)
    )
    for txt_file in all_txt_files_list:
        cleaned_txt_file = cleanup_txt_file(txt_file,processed_dir)
        all_txt_files_list.append(cleaned_txt_file)

    return all_txt_files_list


if __name__ == "__main__":
    """
    docx_path = os.path.join(DOCX_DIR_DUMMY,'sub-a043_transcriptie_MCA.docx')
    interim_dir = PRECLEANTEXT_DIR_DUMMY
    convert_docx_to_txt(docx_path, interim_dir)
    txt_path_in = os.path.join(interim_dir, 'sub-a043_transcriptie_MCA_preclean.txt')
    processed_dir = TEXT_DIR_DUMMY
    cleanup_txt_file(txt_path_in, processed_dir)
    """
    interim_dir = PRECLEANTEXT_DIR
    processed_dir = TEXT_DIR
    preprocess_IANSA_transcripts_startfromtxtfile(interim_dir, processed_dir)

