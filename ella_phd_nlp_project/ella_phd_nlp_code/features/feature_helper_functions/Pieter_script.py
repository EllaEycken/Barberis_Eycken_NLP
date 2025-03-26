#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:35:55 2024

@author: pieter
"""

import pandas as pd
import os
import numpy as np
from docx import Document


# make definitions for reading docx files, listing docx files, removing corrections which have not been said
def read_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def list_doc_files(folder_path):
    doc_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".docx"):
            doc_files.append(os.path.join(folder_path, file))
    return doc_files


def remove_items_between_brackets(words):
    new_words = []
    for word in words:
        if '(' not in word and ')' not in word and not word.endswith('*g'):
            new_words.append(word)
    return new_words


# path specifications
path_input = "L:/GBW-0128_Brain_and_Language/Aphasia/IANSA_study/Papers/2024_VLSM_NS (in progress - cortex SI)/transcripts"
path_output = "L:\\GBW-0128_Brain_and_Language\\Aphasia\\IANSA_study\\07_Analyses\\Tags_from_transcripts\\PDT_interspeech\\count_errors.xlsx"

# search items and their respective columnNames
items = ['*n', '*v', '*a', '*h', '*x', '*F', '*Fa', '*Fo', '*Fs', '*Ft', '*s', '*gr', '*p', '*g', '*D', '*Dw', '*Dk',
         'rep']
items_normalized = list()
for i in items:
    items_normalized.append((i + '_normalized'))

# %%
# collect all docx files
files = list_doc_files(path_input)

# make empty array to store the counts of all search items (i.e., matrix size items x files(=participants))
counts = np.zeros((len(files), len(items)))

# empty list to store the number of words
n_words = np.zeros((len(files), 1))

# subject
subject = list()

# empty list to stor the participant name
for f in range(len(files)):  # loop over all files (i.e., loop over all participants)
    subject.append(files[f].split('\\')[-1].split('.')[0].split('_')[0])

    text = read_docx(files[f])  # read text

    words = text.split()  # count numbner of words (with corrections)
    words_spoken = remove_items_between_brackets(words)  # remove corrections
    n_words[f, 0] = len(words_spoken)

    ## count errors
    for i in range(len(items)):  # loop over all possible search items
        if items[i] == 'rep':
            for w in range(len(words)):
                if w > 0.5:
                    if words[w] == words[w - 1]:
                        counts[f, i] = counts[f, i] + 1
        else:
            for w in words:
                if w.endswith(items[i]):
                    counts[f, i] = counts[f, i] + 1

counts_normalized = counts / n_words

sub_nwords = pd.DataFrame(subject, columns=['subject'])
sub_nwords['n_words'] = n_words
counts = pd.DataFrame(counts, columns=items)
counts_normalized = pd.DataFrame(counts_normalized, columns=items_normalized)
result = pd.concat([sub_nwords, counts, counts_normalized], axis=1)
result = result.sort_values('subject')
result.reset_index(drop=True)

result.to_excel(path_output)



