""" Preprocess transcripts from IANSA dataset (Starting from TEXT FILES) """
""" Use this preprocessing file if the transcripts are TXT-files and not docx.
    Mostly used if you want to use non-annotated transcripts. 
    
    Note that the original text files will now be stored in the INTERIM direction as a starting point (so the raw folder is empty).
    Note also that the original text file names should be sub-XX_nameTask.
    Note that annotations are NOT fixed because it assumes there were none. 
    
    """

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



def cleanup_txt_file(txt_path_in):
    """
    Helper function to clean up a text file (NO new file is created!)

    Note that the text_file names are expected to be sub-X_nameTask (so exactly as the processed text file should be called,
    except for the story files that will be merge into a narrative text file in the preprocess_IANSA function (next step))


    :param txt_path_in: the text file path that must be cleaned up
    :return: returns a string of the original text file with the following changes
        5) Remove ggg, <spk, and xxx
        8) Remove direct speech annotations: :"..."
        9) Clean text formatting: Replace multiple spaces with a single space, Replace double punctuation with single,
         Remove  whitespace in between paragraphs, Add whitespace after '.' if necessary

    """

    ## Read original text file
    try:
        with open(txt_path_in, 'r', encoding="utf-8") as infile:  # means: if encoded by Python (utf-8)
            lines = infile.readlines()
    except UnicodeDecodeError:
        with open(txt_path_in, 'r', encoding="cp1252") as infile:  # means: if encoded by other Windows program (cp1252)
            lines = infile.readlines()

    if not lines:
        raise ValueError("The input file is empty")

    cleaned_fragments = []

    ## Clean the text file (not making a new text file, happens in preprocess function)
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
            cleaned_fragments.append(line)

    return " ".join(cleaned_fragments)


def preprocess_IANSA_transcripts_startfromtxtfile(interim_dir, processed_dir):
    """
    Preprocess ALL IANSA transcripts (using the helper function cleanup_txt_file)
    - Rename ALL cleaned files to:
        sub-XX_transcriptie_task.txt
    - Merge story files into:
        sub-XX_transcriptie_narrative.txt

    :param interim_dir: the interim directory where all the ***ORIGINAL*** TEXT files were stored
    :param processed_dir: the processed directory path where the cleaned TEXT files will be saved
    :return: a list of all the CLEANED text files in the interim directory
    """
    all_txt_files = (
        glob.glob(interim_dir + f"{os.path.sep}sub-a[0-9]*")
        #  the 'sub-a*' looks for all files starting with sub-a
        #  [0-9] means 'any digit and doesn't matter how many digits;
        #  * means 'doesn't matter what comes after this'
        + glob.glob(interim_dir + f"{os.path.sep}sub-b[0-9]*")
        + glob.glob(interim_dir + f"{os.path.sep}sub-c[0-9]*")
    )

    narrative_groups = {}
    other_files = []

    # --- Separate story vs non-story ---
    for txt_path in all_txt_files:
        filename = os.path.basename(txt_path)

        # Detect all possible story types
        if "_story" in filename:
            subject_id = filename.split("_story")[0]

            if subject_id not in narrative_groups:
                narrative_groups[subject_id] = []

            narrative_groups[subject_id].append(txt_path)
        else:
            other_files.append(txt_path)

    cleaned_files = []

    # --- Process non-story files normally ---
    for txt_file in other_files:
        original_filename = os.path.basename(txt_file)
        if "_patientonlyU" in original_filename:
            subject_id = original_filename.split("_")[0]
            task_part = original_filename.split("_")[2]
        else:
            subject_id, task_part = original_filename.split("_", 1)

        output_filename = f"{subject_id}_transcriptie_{task_part}"
        output_path = os.path.join(processed_dir, output_filename)

        cleaned_text = cleanup_txt_file(txt_file)

        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(cleaned_text)

        cleaned_files.append(output_path)

    # --- Handle story files ---
    for subject_id, file_list in narrative_groups.items():

        output_filename = f"{subject_id}_transcriptie_narrative.txt"
        output_path = os.path.join(processed_dir, output_filename)

        merged_text = "".join(  # joins the cleaned text files from story_stroke and story_narrative
            cleanup_txt_file(txt_file)
            for txt_file in sorted(file_list)  # if only one kind of story, just adds that to the merged text
        )

        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.write(merged_text)

        cleaned_files.append(output_path)

    return cleaned_files





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

