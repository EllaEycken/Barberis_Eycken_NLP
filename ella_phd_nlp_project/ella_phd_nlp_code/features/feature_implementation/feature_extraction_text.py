"""Extract linguistic features from text-files
The following features are extracted:
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
    - Word Repetition

"""

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
    # MILTENBURG_MODEL_PATH,
    # NAME_AGREEMENT_PATH,
    TEXT_DIR_DUMMY, # TODO: change this if all is ready!
    annot_unintelligible_word, annots_phonemic_paraphasia, annot_herneming, annots_semantic_paraphasia,
    annots_dialect, annot_neologism, annot_filled_pause, annot_grammatic_error, annot_foreign_language,
    annot_discourse_particle, annot_aborted_word_or_sound
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

from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_helper_functions.helper_extraction_text import *


text_list = read_transcripts(TEXT_DIR_DUMMY)  # TODO: change this if all is ready!


""" SET UP NLP"""
## Call NLP
# epi = epitran.Epitran("nld-Latn")
nlp = spacy.load("nl_core_news_lg")
# Apply my custom fixer
fixer = FixNlpPipeline(nlp)
fixer.fix_nlp_pipeline()


""" SEMANTICS """
class Semantics(object):
    def __init__(self):
        pass

    def semantic_paraphasias(self):
        """
        Calculate the proportion of semantic paraphasias in the transcripts in the directory (self)
        DEF: Substitution of content words for (un)related content words ​(Casilio et al., 2019)​,
        calculated as a proportion of the total number of words.
        self = text_directory
        :return: proportion of semantic paraphasias in the transcript
        """
        prop_sem_pars_list = list()
        list_of_transcripts = read_transcripts(self)
        for transcript in list_of_transcripts:
            cleaned_transcript = CleanTranscript.clean_transcript_for_token_counting(
                transcript)  # see helper function to clean transcripts for token counting
            cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

            doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
            sem_par_list = list()

            for token in doc:
                if any(annotation in str(token) for annotation in annots_semantic_paraphasia):
                   sem_par_list.append(token)

            total_sem_pars = len(sem_par_list)
            total_words = TokenCounter.total_number_of_words(transcript)
            prop_sem_pars = float(total_sem_pars) / total_words
            prop_sem_pars_list.append(prop_sem_pars)

        return prop_sem_pars_list


""" PHONOLOGY """
class Phonology(object):
    def __init__(self):
        pass

    def phonemic_paraphasias(self):
        """
        Calculate the proportion of phonemic paraphasias in the transcripts in the directory (self)
        DEF: Phoneme deletion, insertion, substitution or transposition (Casilio et al., 2019; Vermeulen et al., 1989),
        calculated as a proportion of the total number of words.
        self = text_directory
        :return: proportion of phonemic paraphasias in the transcript
        """
        prop_phon_pars_list = list()
        list_of_transcripts = read_transcripts(self)
        for transcript in list_of_transcripts:
            cleaned_transcript = CleanTranscript.clean_transcript_for_token_counting(
                transcript)  # see helper function to clean transcripts for token counting
            cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

            doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
            phon_par_list = list()

            for token in doc:
                if any(annotation in str(token) for annotation in annots_phonemic_paraphasia):
                    # TODO: switch to more added phon paraphasias if several phon errors in one word should be counted as multiple phon paraphasias
                   phon_par_list.append(token)

            total_phon_pars = len(phon_par_list)
            total_words = TokenCounter.total_number_of_words(transcript)
            prop_phon_pars = float(total_phon_pars) / total_words
            prop_phon_pars_list.append(prop_phon_pars)

        return prop_phon_pars_list


    def neologisms(self):
        """
        Calculate the proportion of neologisms in the transcripts in the directory (self)
        DEF: Nonwords, i.e., word forms that are not real words (Casilio et al., 2019),
        calculated as a proportion of the total number of words.
        self = text_directory
        :return: proportion of neologism in the transcript
        """
        prop_neologisms_list = list()
        list_of_transcripts = read_transcripts(self)
        for transcript in list_of_transcripts:
            cleaned_transcript = CleanTranscript.clean_transcript_for_token_counting(
                transcript)  # see helper function to clean transcripts for token counting
            cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

            doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
            neologism_list = list()

            for token in doc:
                if annot_neologism in str(token):
                   neologism_list.append(token)

            total_neologisms = len(neologism_list)
            total_words = TokenCounter.total_number_of_words(transcript)
            prop_neologisms = float(total_neologisms) / total_words
            prop_neologisms_list.append(prop_neologisms)

        return prop_neologisms_list



""" LEXICAL """
class Lexical(object):
    def __init__(self):
        pass




""" GRAMMATICAL """


""" FLUENCY """




"""RUNNING THE FUNCTIONS"""
if __name__ == "__main__":
    text_directory = TEXT_DIR_DUMMY
    # Semantics.semantic_paraphasias(text_directory)
    # Phonology.phonemic_paraphasias(text_directory)
    Phonology.neologisms(text_directory)
