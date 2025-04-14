""" Helper functions to extract linguistic features from text-files """

""" IMPORT STATEMENTS """

import statistics
import math

import re
import epitran
import nltk
import pandas as pd
import spacy
from spacy.language import Language


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
text_list = read_transcripts(TEXT_DIR_DUMMY)  # TODO: change this if all is ready!

""" NLP HELPER CLASS """

class FixNlpPipeline:
    """
    Fix NLP pipeline for Dutch
    """

    def __init__(self, nlp):
        self.nlp = nlp
        # Register and add components to the pipeline
        self.fix_nlp_pipeline()

    @Language.component("merge_s_constructions")
    # Why is this needed:
    # SpaCy v3+ needs pipeline components to be registered with @Language.component.
    # You then add them by string name using nlp.add_pipe("your_name").
    def merge_s_constructions(doc):
        with doc.retokenize() as retokenizer:
            i = 0
            while i < len(doc) - 1:
                token = doc[i]
                next_token = doc[i + 1]

                # Match: "'s" followed by a lowercase word (e.g. 's morgens, 's avonds)
                if token.text == "'s" and next_token.is_lower:
                    span = doc[i:i + 2]  # Create a span of both tokens
                    attrs = {
                        "pos": "ADV",  # Universal POS tag
                        "tag": "BW"  # Dutch-specific tag for adverbs
                    }
                    retokenizer.merge(span, attrs=attrs)
                    # After merging, the next token has become part of the current one,
                    # so skip to the token after the merged one
                    i += 1
                i += 1
        return doc

    @Language.component("fix_tussentaal")
    def fix_tussentaal_tags(doc):
        for token in doc:
            if token.text.lower() in {"gij", "ge"}:  # was originally tagged as noun
                token.tag_ = "VNW|pers"  # Custom tag
                token.pos_ = "PRON"  # Custom POS (pronoun)
            if token.text.lower() in {"nen", "ne", "ene"}:  # was originally tagged as noun
                token.tag_ = "LID|onbep|stan|agr"
                token.pos_ = "DET"
        return doc

    def fix_nlp_pipeline(self):
        # Add aforementioned methods as pipeline components
        # Only add if not already in pipeline
        if "fix_tussentaal" not in self.nlp.pipe_names:
            self.nlp.add_pipe("fix_tussentaal", last=True)
        if "merge_s_constructions" not in self.nlp.pipe_names:
            self.nlp.add_pipe("merge_s_constructions", last=True)



# epi = epitran.Epitran("nld-Latn")
nlp = spacy.load("nl_core_news_lg")
# Apply my custom fixer
fixer = FixNlpPipeline(nlp)
fixer.fix_nlp_pipeline()




""" OTHER CLASSES"""

class CleanTranscript(object):
    """
    Clean_Transcript Class
    object: a transcript
    """
    def __init__(self):
        pass


    def clean_transcript_for_tagging(self):
        """

        :return: a transcript that is ready to be tagged
        """
        self_str = str(self)  # make string out of transcript
        doc = nlp(self_str)  # read transcript into nlp-doc
        cleaned_tokens = []

        for token in doc:
            if token.is_punct or token.is_space:  # built-in function of token class in Spacy
                continue
            if '*g'in str(token) or '*p' in str(token) or '*a' in str(token):  # annotations
                continue
            if 'Ã' in str(token) or '©' in str(token) or 'â' in str(token) or '€' in str(token) or '¦' in str(token):  # rare characters
                continue
            if '*s' in str(token) or '*S' in str(
                    token):  # make sure to skip the sem paraf (it will count only the function of the normalized version)
                continue
            if '*h' in str(token):  # only keep the 'correct' word 'yyy' from the 'herneming' (xxx-yyy*h) (e.g., ge-geschoten*h)
                match = re.match(r'(\w+)-(\w+)\*h', token.text)
                if match:
                    cleaned_tokens. append(match.group(2))
            else:
                cleaned_tokens.append(str(token))  # this must be turned into a string to later be able to join all
                # elements again
        cleaned_text = ' '.join(cleaned_tokens)  # why join these again: the *h element is seen as a string after the
        # manipulation. It must however be seen as a token again, and Spacy then requires to read the transcript again
        # into an NLP-readable form (you cannot 'create spacy tokens').

        return cleaned_text



class TokenCounter(object):
    """
    Token_Counter Class
    object: a transcript
    """
    def __init__(self):
        pass

    def total_number_of_tokens(self):
        """

        :return: ALL tokens, including punctuation (excluding whitespace: is the tokenizer splitter)

        Notes:
        - suffixes or prefixes are kept APART (e.g., 's avonds = 's + avonds)
        - abbreviations are kept TOGETHER
        """
        self_str = str(self)  # make string out of transcript
        doc = nlp(self_str)  # read transcript into nlp-doc
        token_count = len(doc)  # (doc is immediately split into tokens, based on whitespaces)

        return token_count

    def total_number_of_words(self):
        """

        :return: ALL words, including non-words, phonemic language errors, repetitions, minimal responses,
        comments and stereotypes, in accordance with ​Boxum et al. (2013)​ and ​Vandenborre et al. (2018)​.

        Notes:
        - excludes punctuation!
        """
        self_str = str(self)  # make string out of transcript

        # Step 1: Remove any utterance inside brackets ()
        cleaned_self_str = re.sub(r'\(.*?\)', '', self_str)
        # why: e.g., in semantic paraphasias, now only count paraphasia and not normalized version

        # step 2: Tokenize cleaned text string
        doc = nlp(cleaned_self_str)  # read transcript into nlp-doc
        word_count = 0

        for token in doc:
            if token.is_punct or token.is_space:  # built-in function of token class in Spacy
                continue

            word_count += 1

        return word_count

    # def total_number_of_word_types(self):



class POSTagger(object):
    """
    POS_Tagger class
    object: a transcript

    # Tags list Spacy: https://spacy.io/models/nl --> see 'label scheme' --> tagger
        # Glossary of Spacy tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
        # https://github.com/UniversalDependencies/UD_Dutch-Alpino/blob/master/stats.xml
        # https://github.com/rug-compling/Alpino/blob/master/AlpinoUserGuide.pdf

    """
    def __init__(self):
        pass

    # def show_tag_types(self):  # TODO: make this!
    # LET, N,
    # SPEC|vreemd = ja*p
    # N |eigen|ev|basis|zijd|stan = uh*g
    # LET = .

    def fix_tags(doc): # todo: remove this function
        for token in doc:
            if token.text.lower() in {"gij", "ge"}:
                token.tag_ = "VNW|pers"  # Custom tag
                token.pos_ = "VNW"  # Custom POS (pronoun)
            if token.text.lower() in {"nen", "ne"}:
                token.tag_ = "LID|onbep|stan|agr"
                token.pos_ = "LID"

        return doc

    def tag_list(self, tag_type):
        cleaned_self = CleanTranscript.clean_transcript_for_tagging(self)  # see helper function to clean transcripts for tagging
        cleaned_self_str = str(cleaned_self)  # make string out of transcript
        doc = nlp(cleaned_self_str)  # read transcript into nlp-doc
        tag_list = []

        for token in doc:  # append all tokens with the specific tag (tag_type) to a list
            tagged_token = token.tag_  # tagged_token = 'tag_type|other information on tage_type...
            if tagged_token[0] == tag_type:
                tag_list.append(token)

        return tag_list


    def tag_count(self, tag_type):
        tag_list = self.tag_list(tag_type)
        tag_count = len(tag_list)

        return tag_count


    def tag_rate(self, tag_type):
        tag_count = self.tag_count(tag_type)
        tag_rate = tag_count / TokenCounter.total_number_of_words(self)
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
    POSTagger.tag_list(text,'N')
