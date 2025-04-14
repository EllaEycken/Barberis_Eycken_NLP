""" Preprocess transcripts from IANSA dataset """

import os
import re
import glob
from dataclasses import dataclass
from docx import Document  # first install python-docx
from typing import List

import datasets  # install it first: is a package

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    DOCX_DIR_DUMMY, PRECLEANTEXT_DIR_DUMMY, TEXT_DIR_DUMMY)  # TODO: swap for non-dummy directories once in order


## Create a helper function that converts word docx to a text file
def convert_docx_to_txt(docx_path, interim_dir):
    """
    Helper function to convert a docx file to txt file
    :param docx_path: the word document path
    :param interim_dir: the interim directory path where the txt file must be temporarily saved
    :return: the text file in the interim directory
    """

    # Load the Word document
    doc = Document(docx_path)
    doc_name_subparts = os.path.splitext(os.path.basename(docx_path))[0].split('_')
    # 'os.path.splitext(os.path.basename(docx_path))[0]' extracts the name of the docx_path (all without the .docx
    # extension) so that it can later be used as the name for the text file
    doc_correct_name = str
    if doc_name_subparts[-1] != 'transcriptie':
        doc_correct_name = '_'.join(doc_name_subparts)
    if doc_name_subparts[-1] == 'transcriptie':
        doc_name_subparts[-2],doc_name_subparts[-1] = doc_name_subparts[-1],doc_name_subparts[-2]
        doc_correct_name = '_'.join(doc_name_subparts)

    # Set the name of the text path
    txt_path = os.path.join(
        interim_dir, ".".join(
        ["_".join([str(doc_correct_name),'preclean']),
         'txt']))
    # txt_path = os.path.join(interim_dir, ".".join([str(doc_correct_name),'txt']))


    # Open the text file in write mode
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        # Iterate through each paragraph in the document
        for para in doc.paragraphs:
            # Write the paragraph text to the text file
            txt_file.write(para.text + '\n')

    return txt_path


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
        5) Clean text formatting: Replace multiple spaces with a single space, Replace double punctuation with single,
         Remove  whitespace in between paragraphs, Add whitespace after '.' if necessary

    """
    ## Choose the name for the cleaned text file
    txt_name_subparts = os.path.splitext(os.path.basename(txt_path_in))[0].split('_')
    # 'os.path.splitext(os.path.basename(txt_path_in))[0]' extracts the name of the txt_path (all without the .txt
    # extension)
    txt_correct_name = '_'.join(txt_name_subparts[:-1])
    # make sure you only keep the 'sub-XXX', 'transcriptie' and 'NAME TASK' (so discard the last 'preclean' part) and
    # join them
    txt_path_out = os.path.join(
        processed_dir, ".".join(
            [txt_correct_name, 'txt']))


    with (open(txt_path_in, 'r') as infile, open(txt_path_out, 'w') as outfile):
        # 'r' = open the txt_path_in file for reading, 'w' = open the txt_path_out for writing
        # Note: encoding as utf-8 is needed as Python= , encoding = 'utf-8'

        # first_line = infile.readline()
        # if not first_line:
        #    raise ValueError("The input file is empty")

        lines = infile.readlines()
        if not lines:
            raise ValueError("The input file is empty")

        for line in lines:
            # Remove lines starting with 'Item', 'A:', 'Weekend', or 'Stroke'
            if line.startswith('CAT'):
                continue
            if line.startswith('ANTAT'):
                continue
            if line.startswith('MCA'):
                continue
            if line.startswith('Main'):  # from 'main concept analysis'
                continue
            if line.startswith('Set'):
                continue
            if line.startswith('Item'):
                continue
            if line.startswith('Oefenitem'):
                continue
            if line.startswith('Personal'): # from 'personal narrative'
                continue
            if line.startswith('Weekend'):
                continue
            if line.startswith('Stroke'):
                continue
            # doesn't: work: any(line.startswith(prefix) for prefix in ['ANTAT I', 'MCA transcriptie', 'Set', 'Item', 'Oefenitem',
                                                         # 'Weekend', 'Stroke']):
                # continue

            # Remove 'B:' from the beginning of lines, if present
            if line.startswith('A:'):
                continue
            if line.startswith(' A:'):
                continue
            if line.startswith('B:'):
                line = line[2:]  # Remove 'B:'


            ## Remove abundant text
            line = line.replace('ggg', '').replace('<spk>', '').replace('xxx', '')
            # Remove coughs/laughs (transcribed as 'ggg') from transcript line
            # Remove <spk> transcription for speaker changes from transcript line
            # replace 'xxx' words

            ## Replace [] in [....] statements UNLESS the [] starts with A:, then remove it completely
            # Check if the line starts with 'A:'
            line_splitted = line.split('.')
            line_splitted_fixed = list()
            for sentence in line_splitted:
                if sentence.startswith('[A:') or sentence.startswith(' [A:'):
                    sentence = re.sub(r'\[.*?\]', '', sentence)
                    # swap the [...] statement with a whitespace
                else: # else, just remove the [ and ] but not the content (then it's the speaker's utterances)
                    sentence = sentence.replace('[', '')
                    sentence = sentence.replace(']', '')
                line_splitted_fixed.append(sentence)
            line = '. '.join(line_splitted_fixed)

            ## Remove dialect words and replace them with their normalized versions
            line = re.sub(r'(\w+)\*D\s*\(\s*(.*?)\s*\)', lambda m: m.group(2), line)
            line = re.sub(r'(\w+)\*Dw\s*\(\s*(.*?)\s*\)', lambda m: m.group(2), line)
            line = re.sub(r'(\w+)\*Dk\s*\(\s*(.*?)\s*\)', lambda m: m.group(2), line)
            # r'(...)= regex pattern (regular expression)
            # (\w+) captures a word (alphanumeric characters): here the dialect word that ends with *D.
            # \*D matches the *D annotation. (Dw = dialectische woorden, Dk = dialectische klanken)
            # \(\s*(.*?)\s*\) captures the normalized word within parentheses, allowing for optional whitespace.
            #       \(:
            #           This matches the literal opening parenthesis (. In regex, parentheses are special characters
            #           used for grouping, so to match an actual parenthesis,
            #           you need to escape it with a backslash (\).
            #       \s*:
            #           \s matches any whitespace character (spaces, tabs, etc.).
            #           The * quantifier means "zero or more" occurrences of the preceding element.
            #           So, \s* matches any amount of whitespace (including none) right after the opening parenthesis.
            #       (.*?):
            #           The outer parentheses () create a capturing group, which allows you to extract the
            #           part of the string that matches this pattern.
            #           . matches any character except a newline.
            #           *? is a lazy quantifier that matches "zero or more" of the preceding element
            #           (in this case, any character), but it will stop matching as soon as it finds a match
            #           for the next part of the regex (the closing parenthesis). This is why it’s "lazy";
            #           it tries to match as little as possible.
            #           Therefore, .*? captures everything between the parentheses,
            #           but stops at the first closing parenthesis it encounters.
            #       \s*:
            #           Again, this matches any amount of whitespace (including none)
            #           just before the closing parenthesis.
            #       \):
            #           This matches the literal closing parenthesis ), and like the opening parenthesis,
            #           it is escaped to match the character literally.
            #           the re module provides functions to work with regex. eg 'resub()' allows to replace patterns.
            # the 'lambda m: m.group(2)' in the re.sub function replaces the entire match with the normalized word.
            #       = lambda function in Python is a small, anonymous function defined with the lambda keyword.
            #       It's often used for short, throwaway functions that are not intended to be reused elsewhere.
            #       lambda function lambda m: m.group(2) takes a match object m as input and
            #       returns the second captured group (the normalized word) from the regex match.
            #       Regex Match Object:
            #           When re.sub() finds a match in the target string based on the regex pattern,
            #           it creates a match object. This object contains information about the match,
            #           including the entire matched string and any captured groups.
            #       Captured Groups:
            #           The parentheses in your regex pattern define capturing groups.
            #           For example, in the pattern (\w+)\*D\s*\(\s*(.*?)\s*\), there are two capturing groups:
            #           Group 1: (\w+) (the dialect word)
            #           Group 2: (.*?) (the normalized word)
            #       Passing the Match Object:
            #           When you use a lambda function in re.sub(), the match object is passed as an argument
            #           to the lambda function. Inside the lambda function, you can access the captured groups
            #           using the group() method.

            ## Fix annotation mistakes
            line = line.replace('oei', 'oei*p')
            line = line.replace('fff', 'fff*a')

            ## Clean text formatting
            line = re.sub(r'\s+', ' ', line)  # Replace multiple spaces with a single space
            line = line.replace('. .', '.')  # Replace double punctuation with single
            line = line.strip()  # Strip leading/trailing whitespace
            line = re.sub(r'\.(?!\s)', '. ', line)  # Add whitespace after '.' if necessary


            # Write the modified line to the output file, IF the line is not empty
            if line:
                outfile.write(line)

    return txt_path_out


def preprocess_IANSA_transcripts(raw_dir,interim_dir, processed_dir):
    """
    Preprocess ALL IANSA transcripts (using the helper functions convert_docx_to_txt and cleanup_txt_file)
    :param raw_dir: take the raw directory where all the raw WORD files are located
    :param interim_dir: the interim directory where all the interim PRECLEAN TEXT files will be saved
    :param processed_dir: the processed directory path where the cleaned TEXT files will be saved
    :return: CLEANED text files in the interim directory
    """
    all_txt_files_list = list()
    all_docx_files_list = (
        glob.glob(raw_dir + f"{os.path.sep}sub-a[0-9]*")  # docx from aphasia patients
        #  the 'sub-a*' looks for all files starting with sub-a
        #  [0-9] means 'any digit and doesn't matter how many digits;
        #  * means 'doesn't matter what comes after this'
        + glob.glob(raw_dir + f"{os.path.sep}sub-b[0-9]*") # docx from control patients (comm partner)
        + glob.glob(raw_dir + f"{os.path.sep}sub-c[0-9]*")  # docx from control patients (non-related)
    )
    for docx_file in all_docx_files_list:
        txt_file = convert_docx_to_txt(docx_file, interim_dir)
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
    raw_dir = DOCX_DIR_DUMMY
    interim_dir = PRECLEANTEXT_DIR_DUMMY
    processed_dir = TEXT_DIR_DUMMY
    preprocess_IANSA_transcripts(raw_dir, interim_dir, processed_dir)

