""" Preprocess transcripts from IANSA dataset """

import os
import re
import glob
from dataclasses import dataclass
from docx import Document  # first install python-docx
from typing import List

import datasets  # install it first: is a package

from ella_phd_nlp_project.ella_phd_nlp_code.constants import TEXT_DIR_DUMMY, INTERIM_DIR  # TODO: swap for TEXT_DIR once in order


## Create a helper function that converts word docx to a text file
def convert_docx_to_txt(docx_path, interim_dir):
    """
    Convert docx file to txt file
    :param docx_path: the word document path
    :param interim_dir: the interim directory path where the txt file must be temporarily saved
    :return: the text file in the interim directory
    """

    # Load the Word document
    doc = Document(docx_path)
    doc_name_subparts = os.path.splitext(os.path.basename(docx_path))[0].split('_')
    doc_correct_name = str
    if doc_name_subparts[-1] != 'transcriptie':
        doc_correct_name = '_'.join(doc_name_subparts)
    if doc_name_subparts[-1] == 'transcriptie':
        doc_name_subparts[-2],doc_name_subparts[-1] = doc_name_subparts[-1],doc_name_subparts[-2]
        doc_correct_name = '_'.join(doc_name_subparts)

    txt_path = os.path.join(interim_dir, ".".join([str(doc_correct_name),'txt']))
    # 'os.path.splitext(os.path.basename(docx_path))[0]' extracts the name of the docx_path (all without the .docx
    # extension) so that it can later be used as the name for the text file

    # Open the text file in write mode
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        # Iterate through each paragraph in the document
        for para in doc.paragraphs:
            # Write the paragraph text to the text file
            txt_file.write(para.text + '\n')

    return txt_path


def cleanup_txt_file(txt_path_in, interim_dir):
    """
    Clean up text files
    :param txt_path: the text file path
    :return: returns the text file with the following changes
        1) Remove the title (first line)
        2) Remove the 'items' lines
        3) Remove the 'A:' lines (= test leader)
        4) Remove the 'B:' parts in the participant lines
        5) Remove  whitespace in between paragraphs
        # TODO: changes for MCA, CAT-NL etc
    """
    output_file = os.path.join(
        interim_dir, ".".join(
            ["_".join([os.path.splitext(os.path.basename(txt_path_in))[0], 'clean']),
                'txt']))

    with open(txt_path_in, 'r') as infile, open(output_file, 'w') as outfile:
        # 'r' = open the txt_path_in file for reading, 'w' = open the txt_path_out for writing

        first_line = infile.readline()
        if not first_line: raise ValueError("The input file is empty")

        for line in infile:
            # Remove lines starting with 'Item', 'A:', 'Weekend', or 'Stroke'
            if line.startswith('ANTAT I') or line.startswith('Item') or line.startswith('A:') or line.startswith('Weekend') or line.startswith('Stroke'):
                continue

            # Remove 'B:' from the beginning of lines, if present
            if line.startswith('B:'):
                line = line[2:]  # Remove 'B:'


            # Remove abundant text
            # Remove coughs/laughs (transcribed as 'ggg') from transcript line
            line = line.replace('ggg','')
            # Remove <spk> transcription for speaker changes from transcript line
            line = line.replace('<spk>','')
            # Remove double spaces
            line = line.replace('  ','')
            # Strip whitespace from the line
            line = line.strip()
            # Add whitespace after '.' if there isn't one
            line = re.sub(r'\.(?!\s)', '. ', line)

            # Write the modified line to the output file, IF the line is not empty
            if line:
                outfile.write(line)

    return output_file

def preprocess_IANSA_transcripts(file_path, interim_dir):
    """
    Preprocess IANSA transcripts
    :param text_dir: take the text directory where all WORD files are located
    :return: CLEANED text files in the interim directory
    """
    all_docx_files_list = (
        glob.glob(file_path + f"{os.path.sep}a[0-9]*")  # docx from aphasia patients
        + glob.glob(file_path + f"{os.path.sep}b[0-9]*") # docx from control patients (comm partner)
        + glob.glob(file_path + f"{os.path.sep}c[0-9]*")  # docx from control patients (non-related)
    )
    for docx_file in all_docx_files_list:
        cleanup_txt_file(convert_docx_to_txt(docx_file, interim_dir),interim_dir)

    return all_docx_files_list


if __name__ == "__main__":
    docx_path = os.path.join(TEXT_DIR_DUMMY,'sub-a070_narrative_transcriptie.docx')
    convert_docx_to_txt(docx_path, interim_dir=INTERIM_DIR)
    txt_path_in = os.path.join(INTERIM_DIR, 'sub-a070_transcriptie_narrative.txt')
    interim_dir = INTERIM_DIR
    cleanup_txt_file(txt_path_in, interim_dir)

