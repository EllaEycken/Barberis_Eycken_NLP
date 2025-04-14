""" Helper functions to extract linguistic features from text-files """

""" IMPORT STATEMENTS """

import statistics
import math

import epitran
import nltk
import pandas as pd
import spacy


from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    # AoA_PATH,
    # CONCRETENESS_PATH,
    # FREQUENCY_PATH,
    # DUTCH_ALPHABET,
    # FAMILIARITY_PATH,
    # NAME_AGREEMENT_PATH,
    TEXT_DIR_DUMMY, # TODO: change this if all is ready!
)

# from masterthesisellalaw.data.preprocess_norms import (
    # read_Norms_AoA,
    # read_Norms_concreteness,
    # read_Norms_familiarity,
    # read_Norms_frequency,
    # read_Norms_nameAgreement,
# )

from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import (
    read_transcripts,
)

# epi = epitran.Epitran("nld-Latn")
nlp = spacy.load("nl_core_news_lg")


text_list = read_transcripts(TEXT_DIR_DUMMY)  # TODO: change this if all is ready!



class Token_Counter(object):
    """
    Token_Counter Class
    object: a transcript
    """
    def __init__(self):
        pass

    def total_number_of_tokens(self):
        self_str = str(self)  # make string out of transcript
        doc = nlp(self_str)  # read transcript into nlp-doc
        tokens = doc._.tokenize
        total_tokens = sum(tokens)
        return total_tokens







class POS_Tagger(object):
    """
    POS_Tagger class
    object: a transcript


    """
    def __init__(self):
        pass

    def show_tag_types(self):  # TODO: make this!


    def tag_list(self, tag_type):
        self_str = str(self)  # make string out of transcript
        doc = nlp(self_str)  # read transcript into nlp-doc
        tag_list = []

        for token in doc:  # append all tokens with the specific tag (tag_type) to a list
            tagged_token = token.tag_
            if tagged_token[1] == tag_type:
                tag_list.append(tagged_token[0])

        return tag_list

    def tag_count(self, tag_type):
        tag_list = self.tag_list(tag_type)
        tag_count = len(tag_list)

        return tag_count

    def tag_rate(self, tag_type):
        tag_count = self.tag_count(tag_type)
        tag_rate = tag_count / len(text_list)
        return tag_rate









"""
- Semantics:
    - semantic paraphasias
- Phonology
    - phonemic paraphasias
    - Neologisms
- Lexical
    - Number of words
    - Brunet's Index
    - Noun Rate
    - Verb Rate
    - Adjective Rate
    - Pronoun Rate
    - Adverb Rate
    - Determiner Rate
    - Conjunction Rate
    - Preposition Rate
    - Particle Rate
    - Content-Function Rate
- Grammatical
    - Mean Length of Utterance
    - Noun-verb Rate
    - Subordinate Clause
    - Syntactic Deviation
- Fluency
    - Filled Pause
    - False Start
    - Word Abandoned
    - Word Repetition"""



for text in text_list:
    doc = nlp(text)
    for token in doc:
        tagged_text = token.tag_
        # Tags list Spacy: https://spacy.io/models/nl --> see 'label scheme' --> tagger
        # Glossary of Spacy tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
        # https://github.com/UniversalDependencies/UD_Dutch-Alpino/blob/master/stats.xml
        # https://github.com/rug-compling/Alpino/blob/master/AlpinoUserGuide.pdf