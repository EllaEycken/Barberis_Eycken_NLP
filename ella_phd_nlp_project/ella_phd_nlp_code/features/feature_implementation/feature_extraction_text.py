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
    - (Syntactic Deviation)
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
    TEXT_DIR,
    # TEXT_DIR_DUMMY,
    modal_lemmas,
    articles_dutch,
    particles_dutch,
    annot_unintelligible_word, annots_phonemic_paraphasia, annot_false_start, annots_semantic_paraphasia,
    annots_dialect, annot_neologism, annot_filled_pause, annot_grammatic_error, annot_foreign_language,
    annot_discourse_particle, annot_aborted_word_or_sound, subordinate_dependencies
)

from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import (
    read_transcripts,
)

from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_helper_functions.helper_extraction_text import *


text_list = read_transcripts(TEXT_DIR)


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
    """ Calculate the proportion of semantic paraphasias in the transcripts in the directory
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
    """Calculate the proportion of phonemic paraphasias in the transcripts in the directory
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
    """ Calculate the proportion of neologisms in the transcripts in the directory
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
    """ Calculate the total number of words in the transcripts in the directory
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
    """ Calculate Brunet's index in the transcripts in the directory
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
    """ Calculate noun rate in the transcripts in the directory
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
    """ Calculate verb rate in the transcripts in the directory
    DEF: Total number of verbs* divided by the total number of words.
    *main verbs (hoofdwerkwoorden): excluding auxilliary and modal verbs
    todo: if possible, append in NLP pipeline that combination of 'worden' and 'gaan' makes 'worden' an auxilliary verb

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
            elif verb.lemma_ in modal_lemmas and verb.pos_ in {"VERB", "AUX"}:  # modal_lemmas is from constants
                continue
            else:  # if not AUX, and not modal verb, it is a main 'VERB': then it's a main verb
                main_verb_list.append(verb)
        main_verb_count = len(main_verb_list)
        main_verb_rate = main_verb_count / TokenCounter(transcript).total_number_of_words()
        verb_rate_list.append(main_verb_rate)

    return verb_rate_list


def adjective_rate(
        file_path: str,
):
    """ Calculate adjective rate in the transcripts in the directory
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
    """ Calculate pronoun rate in the transcripts in the directory
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



def adverb_rate(
        file_path: str,
):
    """ Calculate adverb rate in the transcripts in the directory
    DEF: Total number of adverbs divided by the total number of words.

    :file_path: text_directory
    :return: adverb rates in the transcripts
    """
    adverb_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        adverb_rate = POSTagger(transcript).tag_rate(tag_type = "BW")  # "BW" = for adverbs
        adverb_rate_list.append(adverb_rate)

    return adverb_rate_list



def determiner_rate(
        file_path: str,
):
    """ Calculate determiner rate in the transcripts in the directory
    DEF: Total number of determiners divided by the total number of words.
    DEF: determiners for Dutch includes articles (lidwoorden), numerals (telwoorden), demonstrative and possessive
    pronouns (aanwijzende en bezittelijke voornaamwoorden), and quantifiers (kwantoren).
    Source: https://nl.wikipedia.org/wiki/Determinator_(klasse)
    Note: here we work with POS_ (upos and xpos) as they are more specific than tag_. Moreover, for some words (pronouns)
    the morph_function is needed to capture determiners specifically for Dutch.

    :file_path: text_directory
    :return: determiner rates in the transcripts
    """
    determiner_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        cleaned_text = CleanTranscript(transcript).clean_transcript_for_tagging()  # clean the text
        cleaned_text_str = str(cleaned_text)  # make string out of cleaned transcript

        doc = nlp(cleaned_text_str)  # read transcript into nlp-doc
        determiner_list = []

        for token in doc:  # append all tokens with the specific tag (tag_type) to a list
            if token.pos_ == "DET":  # articles (lidwoorden), quantifiers (kwantoren)
                # Note: 'DET' is the UPOS (universal Part of Speech), which is multilingual but
                # doesn't grasp all determiners specifically for Dutch,
                # namely, the UPOS tag focuses on articles (lidwoorden) en quantifiers
                # while it doesn't tag pronouns as determiners
                determiner_list.append(token)
            elif token.pos_ == "PRON":  # pronouns (aanwijzende vnwen)
                # Check morphological features for demonstrative or possessive or indicator use
                # possessive use = possessive pronoun (bezittelijk vnw), eg 'mijn, hun'...
                # demonstrative use = demonstrative pronoun (aanwijzend vnw), eg 'die, dat'
                # indicator use = quantifier (kwantor), eg 'alle, elke' (wordt ook soms POS tag DET gegeven ipv pronoun)
                if "Poss=Yes" in token.morph or "PronType=Dem" in token.morph or "PronType=Ind" in token.morph:
                    determiner_list.append(token)
            elif token.pos_ == "NUM":  # numerals (telwoorden)
                # Check for cardinals or ordinals used as determiners
                if token.dep_ == "det":
                    determiner_list.append(token)

        determiner_count = len(determiner_list)
        determiner_rate = determiner_count/TokenCounter(transcript).total_number_of_words()
        determiner_rate_list.append(determiner_rate)

    return determiner_rate_list



def conjunction_rate(
        file_path: str,
):
    """ Calculate conjunction rate in the transcripts in the directory
    DEF: Total number of conjunctions divided by the total number of words.
    DEF: this includes coordinating conjunctions (nevenschikkende voegwoorden) and subordinating conjunctions
    (onderschikkende voegwoorden).

    :file_path: text_directory
    :return: conjunction rates in the transcripts
    """
    conjunction_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        cleaned_text = CleanTranscript(transcript).clean_transcript_for_tagging()  # clean the text
        cleaned_text_str = str(cleaned_text)  # make string out of cleaned transcript

        doc = nlp(cleaned_text_str)  # read transcript into nlp-doc
        conjunctions_list = []

        for token in doc:  # append all tokens with the specific tag (tag_type) to a list
            if token.pos_ in {"CCONJ", "SCONJ"}:  # append tokens with the function of coordinating conjunctions (cconj)
                # or subordinating conjunctions (sconj) to the list
                conjunctions_list.append(token)

        conjunction_count = len(conjunctions_list)
        conjunction_rate = conjunction_count/TokenCounter(transcript).total_number_of_words()
        conjunction_rate_list.append(conjunction_rate)

    return conjunction_rate_list



def preposition_rate(
        file_path: str,
):
    """ Calculate preposition rate in the transcripts in the directory
    DEF: Total number of prepositions divided by the total number of words.

    :file_path: text_directory
    :return: preposition rates in the transcripts
    """
    preposition_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        preposition_rate = POSTagger(transcript).tag_rate(tag_type = "VZ")  # "VZ" = for prepositions (voorzetsels)
        # Note that for universal tagging (non-Dutch specific), I could also use token.pos_ == "ADP"
        preposition_rate_list.append(preposition_rate)

    return preposition_rate_list



def particle_rate(
        file_path: str,
):
    """ Calculate the particle rate in the transcripts in the directory
    DEF: Total number of particles* divided by the total number of words.
    *The following tokens are seen as particles: [eigenlijk, ja, nee, toch, nou, wel, hoor, he, inderdaad, eens,
    even, allez/allee, oei, goh, ze, se, enzo, enzovoort, zenne, enfin] + other interjections specific to the context
    todo: is 'goeiemorgen' and 'goeiemiddag' a particle?

    :file_path: text_directory
    :return: particle rates in the transcripts
    """
    particle_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # see helper function to clean transcripts for token counting
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
        particle_list = list()

        for token in doc:
            if annot_discourse_particle in str(token):
               particle_list.append(token)
            elif token in particles_dutch:
                particle_list.append(token)


        particle_count = len(particle_list)
        total_words = TokenCounter(transcript).total_number_of_words()
        particle_rate = particle_count/ total_words
        particle_rate_list.append(particle_rate)

    return particle_rate_list



def content_function_ratio(
        file_path: str,
):
    """ Calculate content-function ratio in the transcripts in the directory
    DEF: the total number of content words (that express semantic content) divided by total number of function words
    (that express grammatical relationships rather than semantic content).
    - content words include: nouns, lexical verbs (= non-auxiliary), adjectives, adverbs
    - function words include: articles, pronouns, adpositions, conjunctions (coordinating and subordinating),
    auxiliary verbs, particles.
    Source: https://en.wikipedia.org/wiki/Content_word and https://en.wikipedia.org/wiki/Function_word
    IMPLICATION: Measure of lexical proportion (Gordon, 2020).
    - High content-function ratio is characteristic of agrammatic speech (Gordon, 2006; Saffran et al., 1989)
    - Low content-function word ratio is characteristic of empty speech (Gordon, 2006)(Edwards, 2005).

    :file_path: text_directory
    :return: content-function rates in the transcripts

    todo: in Wikipedia, function words also include interjections, expletives and pro-sentences. But seems bit odd. I left them out for now.
    todo: is 'goeiemorgen' and 'goeiemiddag' a particle?
    """
    content_function_ratio_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:

        ## Set empty lists
        content_words_list = list()
        noun_list = list()
        lexical_verb_list = list()
        adjective_list = list()
        adverb_list = list()

        function_words_list = list()
        article_list = list()
        pronoun_list = list()
        preposition_list = list()
        conjunction_list = list()
        auxiliary_verb_list = list()
        particle_list = list()
        # interjection_list = list()  # note: token.pos_ == 'INTJ'
        # expletive_list = list()
        # pro_sentence_list = list()


        ##  Make lists of the content and function words based on POS tagging
        noun_list = POSTagger(transcript).tag_list(tag_type="N")  # "N" = for nouns
        adjective_list = POSTagger(transcript).tag_list(tag_type = "ADJ")  # "ADJ" = for adjectives
        adverb_list = POSTagger(transcript).tag_list(tag_type="BW")  # "BW" = for adverbs
        preposition_list = POSTagger(transcript).tag_list(tag_type="VZ")  # "VZ" = for prepositions (voorzetsels)
        pronoun_list = POSTagger(transcript).tag_list(tag_type="VNW")  # "N" = for nouns
        lexical_verb_list = list()
        auxiliary_verb_list = list()
        verb_list = POSTagger(transcript).tag_list(tag_type="WW")  # "WW" = for verbs
        for verb in verb_list:  # check for each verb whether it's a main verb (then append it to the main verb list) or
            # an auxiliary verb (AUX), then append it to the auxiliary verb list
            pos_verb = verb.pos_  # verb form can be checked with the POS command (pos = universal pos tag, while tag = detailed morphological tag)
            if 'AUX' in str(pos_verb):
                auxiliary_verb_list.append(verb)
            elif verb.lemma_ in modal_lemmas and verb.pos_ in {"VERB", "AUX"}:
                # sometimes the verb is not recognized as an auxiliary verb while it is, then look at the lemma
                # note: modal_lemmas is from constants
                auxiliary_verb_list.append(verb)
            else:  # if not AUX, and not modal verb, it is a main 'VERB': then it's a main verb
                lexical_verb_list.append(verb)


        ##  Make lists of the content and function words based on in text- or other annotations
        cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # see helper function to clean transcripts for token counting
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc

        for token in doc:
            if annot_discourse_particle in str(token):
                particle_list.append(token)
            elif str(token) in particles_dutch:
                particle_list.append(token)
            elif token.pos_ in {"CCONJ", "SCONJ"}:  # append tokens with the function of coordinating conjunctions (cconj)
                        # or subordinating conjunctions (sconj) to the list
                conjunction_list.append(token)
            elif str(token) in articles_dutch:
                article_list.append(token)

        content_words_list = noun_list + lexical_verb_list + adjective_list + adverb_list
        function_words_list = article_list + pronoun_list + preposition_list + conjunction_list + auxiliary_verb_list + particle_list

        content_word_count = len(content_words_list)
        function_word_count = len(function_words_list)
        if function_word_count == 0:  # avoid dividing by 0
            content_function_ratio = 1
        else:
            content_function_ratio = content_word_count / function_word_count

        content_function_ratio_list.append(content_function_ratio)


    return content_function_ratio_list



""" GRAMMATICAL """
def noun_verb_rate(
        file_path: str,
):
    """ Calculate noun-verb rate in the transcripts in the directory
    DEF: Total number of nouns divided by total number of verbs, excluding auxiliaries and modals

    :file_path: text_directory
    :return: noun-verb rates in the transcripts
    """
    noun_verb_rate_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        noun_list = POSTagger(transcript).tag_list(tag_type="N")  # "N" = for nouns
        main_verb_list = list()
        verb_list = POSTagger(transcript).tag_list(tag_type="WW")  # "WW" = for verbs
        for verb in verb_list:  # check for each verb whether it's a main verb (then append it to the list) or an auxilliary verb (AUX), then don't.
            pos_verb = verb.pos_  # verb form can be checked with the POS command (pos = universal pos tag, while tag = detailed morphological tag)
            if 'AUX' in str(pos_verb):
                continue
            elif verb.lemma_ in modal_lemmas and verb.pos_ in {"VERB", "AUX"}:
                # sometimes the verb is not recognized as an auxiliary verb while it is, then look at the lemma
                # note: modal_lemmas is from constants
                continue
            else:  # if not AUX, and not modal verb, it is a main 'VERB': then it's a main verb
                main_verb_list.append(verb)

        noun_count = len(noun_list)
        main_verb_count = len(main_verb_list)
        noun_verb_rate = 0
        if main_verb_count == 0:  # avoid dividing by 0
            noun_verb_rate = 1
        else:
            noun_verb_rate = noun_count/main_verb_count
        noun_verb_rate_list.append(noun_verb_rate)

    return noun_verb_rate_list


def mean_length_utterance(
        file_path: str,
):
    """ Calculate the MLU (mean length of utterance) in the transcripts in the directory
    DEF: Average number of words per utterance, based on the guidelines provided by (Boxum et al., 2013).
    Utterances are split using the helper function Utterance(transcript).split_into_custom_utterances():
    Split transcripts into utterances complying with linguistic criteria
        1) “And”: New utterance unless it’s part of an enumeration or combined action.
        2) Conjunctions: Only separate utterances if used disfluently or non-functionally.
        ( 3) Direct speech: All direct speech after a colon (e.g., He said: "I don’t know.") is treated as a distinct unit.
        Each sentence within the direct quote counts as a separate utterance.)
        4) Embedded utterances: Interjected comments within a sentence are extracted as separate utterances
        (e.g., parentheticals, commas with discourse phrases like "I don't know", etc.).

    :file_path: text_directory
    :return: MLU in the transcripts
    """
    mlu_list = list()
    list_of_transcripts = read_transcripts(file_path)

    for transcript in list_of_transcripts:
        utterances = Utterance(transcript).split_into_custom_utterances()
        list_of_utterance_lengths = []
        if not utterances:
            return 0.0
        for utt in utterances:
            len_utterance = len(nlp(utt))
            list_of_utterance_lengths.append(len_utterance)

        sum_of_utterance_lengths = sum(list_of_utterance_lengths)
        amount_of_utterances = len(list_of_utterance_lengths)
        mlu = sum_of_utterance_lengths/amount_of_utterances

        mlu_list.append(mlu)

    return mlu_list


def subordinate_clauses(
        file_path: str,
):
    """ Calculate the proportion of subordinate clauses in the transcripts in the directory
    DEF: Clause that forms part of and is dependent on a main clause, often introduced by a conjunction.
    Calculated as a proportion of the total number of utterances.
    Definition of a subordinate clause was based on:
    In syntax, a subordinate clause is a dependent clause that cannot stand alone. It usually starts with:
        - Subordinating conjunctions: because, although, since, if, when, unless, while, etc.
        - Relative pronouns: who, which, that
        - Wh- words: what, how, why, where

    :file_path: text_directory
    :return: proportion of subordinate clauses in the transcripts
    """
    prop_subordinate_clauses_list = list()

    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        list_of_utterances = Utterance(transcript).split_into_custom_utterances()
        list_of_subordinate_clauses = list()

        for utt in list_of_utterances:
            utt_doc = nlp(utt)
            for token in utt_doc:
                if token.dep_ in subordinate_dependencies:
                    list_of_subordinate_clauses.append(utt)

        total_subordinate_clauses = len(list_of_subordinate_clauses)
        total_utterances = len(list_of_utterances)
        prop_subordinate_clauses = total_subordinate_clauses/total_utterances
        prop_subordinate_clauses_list.append(prop_subordinate_clauses)

    return prop_subordinate_clauses_list



""" FLUENCY """
def filled_pauses(
        file_path: str,
):
    """ Calculate the proportion of filled pauses in the transcripts in the directory
    DEF: Filler words (e.g., ‘uh’) calculated as a proportion of the total number of words.

    :file_path: text_directory
    :return: proportion of filled pauses in the transcripts
    """
    prop_filled_pauses_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # see helper function to clean transcripts for token counting
        # Note: Why CleanTranscript.clean_transcript_for_token_counting(my_text) doesn't work:
        # clean_transcript_for_token_counting is an instance method,
        # meaning it expects to be called on an object (an instance of the class), not the class itself.
        # So it will throw a TypeError, because self isn't automatically passed in that case.
        # cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # This will make sure the function (clean_transcript_for...) will be calculated on the object (class
        # instance) 'CleanTranscript(transcript)' and not on the class CleanTranscript itself.
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
        filled_pauses_list = list()

        for token in doc:
            if annot_filled_pause in str(token):
                filled_pauses_list.append(token)
            # note: don't use 'if any(annotation in str(token) for annotation in annot_filled_pause):..." because
            # there is only one element in annot_filled_pause, so if you apply the 'any()' function, it will split
            # that one annotation (here '*g') into its components (here '*' and 'g') and look for occurrences of those
            # components in each token.
            # As a result, it will list all tokens with * and all tokens containing the letter 'g' in the filled_pauses_list,
            # overestimating the real amount.

        total_filled_pauses = len(filled_pauses_list)
        total_words = TokenCounter(transcript).total_number_of_words()
        # idem as above regarding the clean_transcripts function
        prop_filled_pauses = float(total_filled_pauses) / total_words
        prop_filled_pauses_list.append(prop_filled_pauses)

    return prop_filled_pauses_list



def false_starts(
        file_path: str,
):
    """ Calculate the proportion of false starts in the transcripts in the directory
    DEF: A word is temporarily interrupted, but the target word is eventually produced (e.g., ca-camera).
    Calculated as a proportion of the total number of words.

    :file_path: text_directory
    :return: proportion of false starts in the transcripts
    """
    prop_false_starts_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # see helper function to clean transcripts for token counting
        # Note: Why CleanTranscript.clean_transcript_for_token_counting(my_text) doesn't work:
        # clean_transcript_for_token_counting is an instance method,
        # meaning it expects to be called on an object (an instance of the class), not the class itself.
        # So it will throw a TypeError, because self isn't automatically passed in that case.
        # cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # This will make sure the function (clean_transcript_for...) will be calculated on the object (class
        # instance) 'CleanTranscript(transcript)' and not on the class CleanTranscript itself.
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
        false_starts_list = list()

        for token in doc:
            if annot_false_start in str(token):
                false_starts_list.append(token)
            # note: don't use 'if any(annotation in str(token) for annotation in annot_false_start):..." because
            # there is only one element in annot_false_start, so if you apply the 'any()' function, it will split
            # that one annotation (here '*h') into its components (here '*' and 'h') and look for occurrences of those
            # components in each token.
            # As a result, it will list all tokens with * and all tokens containing the letter 'g' in the false_starts_list,
            # overestimating the real amount.

        total_false_starts = len(false_starts_list)
        total_words = TokenCounter(transcript).total_number_of_words()
        # idem as above regarding the clean_transcripts function
        prop_false_starts = float(total_false_starts) / total_words
        prop_false_starts_list.append(prop_false_starts)

    return prop_false_starts_list



def word_abandoned(
        file_path: str,
):
    """ Calculate the proportion of abandoned words in the transcripts in the directory
    DEF: Words are abandoned after one or two phonemes; the target word is not produced.
    Calculated as a proportion of the total number of words.

    :file_path: text_directory
    :return: proportion of abandoned words in the transcripts
    """
    prop_word_abandoned_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # see helper function to clean transcripts for token counting
        # Note: Why CleanTranscript.clean_transcript_for_token_counting(my_text) doesn't work:
        # clean_transcript_for_token_counting is an instance method,
        # meaning it expects to be called on an object (an instance of the class), not the class itself.
        # So it will throw a TypeError, because self isn't automatically passed in that case.
        # cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # This will make sure the function (clean_transcript_for...) will be calculated on the object (class
        # instance) 'CleanTranscript(transcript)' and not on the class CleanTranscript itself.
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
        abandoned_words_list = list()

        for token in doc:
            if annot_aborted_word_or_sound in str(token):
                abandoned_words_list.append(token)
            # note: idem note as in filled_pauses() function

        total_abandoned_words = len(abandoned_words_list)
        total_words = TokenCounter(transcript).total_number_of_words()
        # idem as above regarding the clean_transcripts function
        prop_word_abandoned = float(total_abandoned_words) / total_words
        prop_word_abandoned_list.append(prop_word_abandoned)

    return prop_word_abandoned_list



def word_repetition(
        file_path: str,
):
    """ Calculate the proportion of word repetitions in the transcripts in the directory
    DEF: Repetition of a previously used word, calculated as a proportion of the total number of words.

    Note: in the helper.py, the cleanup_transcript functions handle repetitions in that way that consecutive repetitions
    of a word are reduced to a maximum of two occurrences. As such, repetitions are not 'overcounted'.
    Thus, a repetition is seen as 'the observation that a target word has been repeated at least 1 time'. And it is
    this observation alone that is counted, not the amount of repetitions itself.

    :file_path: text_directory
    :return: proportion of word repetitions in the transcripts
    """
    prop_word_repetitions_list = list()
    list_of_transcripts = read_transcripts(file_path)
    for transcript in list_of_transcripts:
        cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # see helper function to clean transcripts for token counting
        # Note: Why CleanTranscript.clean_transcript_for_token_counting(my_text) doesn't work:
        # clean_transcript_for_token_counting is an instance method,
        # meaning it expects to be called on an object (an instance of the class), not the class itself.
        # So it will throw a TypeError, because self isn't automatically passed in that case.
        # cleaned_transcript = CleanTranscript(transcript).clean_transcript_for_token_counting()
        # This will make sure the function (clean_transcript_for...) will be calculated on the object (class
        # instance) 'CleanTranscript(transcript)' and not on the class CleanTranscript itself.
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        doc = nlp(cleaned_transcript_str)  # read transcript into nlp-doc
        word_repetition_list = list()

        for token in doc:
            if annot_repetition in str(token):
                word_repetition_list.append(token)
            # note: idem note as in filled_pauses() function

        total_word_repetitions = len(word_repetition_list)
        total_words = TokenCounter(transcript).total_number_of_words()
        # idem as above regarding the clean_transcripts function
        prop_word_repetitions = float(total_word_repetitions) / total_words
        prop_word_repetitions_list.append(prop_word_repetitions)

    return prop_word_repetitions_list



"""RUNNING THE FUNCTIONS"""
if __name__ == "__main__":
    text_dir = TEXT_DIR
    # semantic_paraphasias(text_dir)
    # neologisms(text_dir)
    # number_of_words(text_dir)
    # brunets_index(text_dir)
    # noun_rate(text_dir)
    # verb_rate(text_dir)
    # adjective_rate(text_dir)
    # pronoun_rate(text_dir)
    # adverb_rate(text_dir)
    # determiner_rate(text_dir)
    # conjunction_rate(text_dir)
    # preposition_rate(text_dir)
    # particle_rate(text_dir)
    # noun_verb_rate(text_dir)
    # filled_pauses(text_dir)
    # false_starts(text_dir)
    # word_abandoned(text_dir)
    # word_repetition(text_dir)
    # content_function_ratio(text_dir)
    # mean_length_utterance(text_dir)
    subordinate_clauses(text_dir)