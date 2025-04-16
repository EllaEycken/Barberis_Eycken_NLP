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


def semantic_paraphasias(
        file_path: str,
):
    """ Calculate the proportion of semantic paraphasias in the transcripts in the directory (self)
    DEF: Substitution of content words for (un)related content words ​(Casilio et al., 2019)​,
    calculated as a proportion of the total number of words.

    :file_path: text_directory
    :return: proportion of semantic paraphasias in the transcripts
    """
    prop_sem_pars_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()  # see helper function to clean transcripts for token counting
        # Note: Why CleanTranscript.clean_transcript_for_token_counting(my_text) doesn't work:
        # clean_transcript_for_token_counting is an instance method,
        # meaning it expects to be called on an object (an instance of the class), not the class itself.
        # So it will throw a TypeError, because self isn't automatically passed in that case.
        # cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # This will make sure the function (clean_transcript_for...) will be calculated on the object (class
        # instance) 'CleanTranscript(transcript)' and not on the class CleanTranscript itself.
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
        sem_par_list = list()

        for token in doc:
            if any(annotation in str(token) for annotation in annots_semantic_paraphasia):
                sem_par_list.append(token)

        total_sem_pars = len(sem_par_list)
        total_words = TokenCounter(transcript).total_number_of_words()
        # idem as above regarding the clean_transcripts function
        prop_sem_pars = float(total_sem_pars) / total_words
        prop_sem_pars_list.append(prop_sem_pars)

    return prop_sem_pars_list




""" PHONOLOGY """


def phonemic_paraphasias(
        file_path: str,
):
    """Calculate the proportion of phonemic paraphasias in the transcripts in the directory (self)
    DEF: Phoneme deletion, insertion, substitution or transposition (Casilio et al., 2019; Vermeulen et al., 1989),
    calculated as a proportion of the total number of words.

    :file_path: text_directory
    :return: proportion of phonemic paraphasias in the transcripts
    """
    prop_phon_pars_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # see helper function to clean transcripts for token counting
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
        phon_par_list = list()

        for token in doc:
            if any(annotation in str(token) for annotation in annots_phonemic_paraphasia):
                # TODO: switch to more added phon paraphasias if several phon errors in one word should be counted as multiple phon paraphasias
                phon_par_list.append(token)

        total_phon_pars = len(phon_par_list)
        total_words = TokenCounter(transcript).total_number_of_words()
        prop_phon_pars = float(total_phon_pars) / total_words
        prop_phon_pars_list.append(prop_phon_pars)

    return prop_phon_pars_list


def neologisms(
        file_path: str,
):
    """ Calculate the proportion of neologisms in the transcripts in the directory (self)
    DEF: Nonwords, i.e., word forms that are not real words (Casilio et al., 2019),
    calculated as a proportion of the total number of words.

    :file_path: text_directory
    :return: proportion of neologism in the transcripts
    """
    prop_neologisms_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # see helper function to clean transcripts for token counting
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
        neologism_list = list()

        for token in doc:
            if annot_neologism in str(token):
               neologism_list.append(token)

        total_neologisms = len(neologism_list)
        total_words = TokenCounter(transcript).total_number_of_words()
        prop_neologisms = float(total_neologisms) / total_words
        prop_neologisms_list.append(prop_neologisms)

    return prop_neologisms_list



""" LEXICAL """
def number_of_words(
        file_path: str,
):
    """ Calculate the total number of words in the transcripts in the directory (self)
    DEF: Total number of words produced, including non-words, phonemic language errors, repetitions,
    minimal responses, comments and stereotypes, in accordance with Boxum et al. (2013) and Vandenborre et al.(2018).

    :file_path: text_directory
    :return: total number of words in the transcript
    """
    nb_of_words_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        total_nb_of_words = TokenCounter(transcript).total_number_of_words()
        nb_of_words_list.append(total_nb_of_words)

    return nb_of_words_list


def brunets_index(
        file_path: str,
):
    """ Calculate Brunet's index
    DEF: Text-length independent measure of lexical diversity. Calculated as  w^(u^−0.165)
    where w = total number of word tokens
    and u = total number of unique word types
    (Parsa et al., 2021).
    Lower values indicate richer speech (Sanborn et al., 2020).

    :file_path: text_directory
    :return: list of floats containing the Brunet's index
    """
    BI_list = []  # define a now still empty list of Brunet Indexes
    BI_constant = float(-0.165)
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        w = TokenCounter(transcript).total_number_of_words()
        u = TokenCounter(transcript).total_number_of_word_types()

        BI = math.pow(w, math.pow(u, BI_constant))  # formula of Burnet's index (math.pow = exponentiation)
        BI_list.append(BI)

    return BI_list


def noun_rate(
        file_path: str,
):
    """ Calculate noun rate
    DEF: Total number of nouns divided by the total number of words.

    :file_path: text_directory
    :return: noun rates in the transcripts
    """
    noun_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        noun_rate = POSTagger(transcript).tag_rate(tag_type = "N")  # "N" = for nouns
        noun_rate_list.append(noun_rate)

    return noun_rate_list


def verb_rate(
        file_path: str,
):
    """ Calculate verb rate
    DEF: Total number of verbs* divided by the total number of words.
    *main verbs (hoofdwerkwoorden)

    :file_path: text_directory
    :return: verb rates in the transcripts
    """
    verb_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        main_verb_list = list()
        verb_list = POSTagger(transcript).tag_list(tag_type = "WW")  # "WW" = for verbs
        for verb in verb_list:  # check for each verb whether it's a main verb (then append it to the list) or an auxilliary verb (AUX), then don't.
            pos_verb = verb.pos_  # verb form can be checked with the POS command (pos = universal pos tag, while tag = detailed morphological tag)
            if 'AUX' in str(pos_verb):
                continue
            else:  # if not AUX, it is 'VERB': then it's a main verb
                main_verb_list.append(verb)
        main_verb_count = len(main_verb_list)
        main_verb_rate = main_verb_count / TokenCounter(transcript).total_number_of_words()
        verb_rate_list.append(main_verb_rate)

    return verb_rate_list


def adjective_rate(
        file_path: str,
):
    """ Calculate adjective rate
    DEF: Total number of adjectives divided by the total number of words.

    :file_path: text_directory
    :return: adjective rates in the transcripts
    """
    adj_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        adj_rate = POSTagger(transcript).tag_rate(tag_type = "ADJ")  # "N" = for nouns
        adj_rate_list.append(adj_rate)

    return adj_rate_list



def pronoun_rate(
        file_path: str,
):
    """ Calculate pronoun rate
    DEF: Total number of pronouns divided by the total number of words.

    :file_path: text_directory
    :return: pronoun rates in the transcripts
    """
    pronoun_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        pronoun_rate = POSTagger(transcript).tag_rate(tag_type = "VNW")  # "N" = for nouns
        pronoun_rate_list.append(pronoun_rate)

    return pronoun_rate_list

""" GRAMMATICAL """


""" FLUENCY """




"""RUNNING THE FUNCTIONS"""
if __name__ == "__main__":
    text_dir = TEXT_DIR_DUMMY
    # semantic_paraphasias(text_dir)
    # neologisms(text_dir)
    # number_of_words(text_dir)
    # brunets_index(text_dir)
    # noun_rate(text_dir)
    # verb_rate(text_dir)
    # adjective_rate(text_dir)
    pronoun_rate(text_dir)