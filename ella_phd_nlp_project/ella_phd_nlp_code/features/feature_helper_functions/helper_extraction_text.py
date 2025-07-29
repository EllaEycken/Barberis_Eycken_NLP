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
    annot_repetition,
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



""" NLP PIPELINE CLASS """

class FixNlpPipeline:
    """
    Fix NLP pipeline for Dutch
    This includes:
    - merge s constructions: all constructions starting with 's should be merged as one token (e.g., 's avonds)
    - fix tussentaal tags: specific tussentaal tags (gij, ge, nen) should have the correct tag (e.g., gij = VNW)


    Note: if I want to add other changes to the pipeline:
    Option 1) Add new language component (with different name)
    Option 2) Append items in existing language component (eg in 'fix_tussentaal'):
        make changes
        THEN: reload the model to start fresh
        nlp = spacy.load("nl_core_news_lg")  # reloads the model cleanly
        fixer = FixNlpPipeline(nlp)
        fixer.fix_NLP_pipeline()

    """

    def __init__(self, nlp):
        self.nlp = nlp
        # Register and add components to the pipeline
        self.fix_nlp_pipeline()

    @Language.component("merge_'s_constructions")
    # Note: Why is this 'language component' needed in the code:
    # SpaCy v3+ needs pipeline components to be registered with @Language.component.
    # You then add them by string name using nlp.add_pipe("your_name").
    def merge_s_constructions(doc):
        """
        merge s constructions: all constructions starting with 's should be merged as one token (e.g., 's avonds)
        TODO: doesn't work properly, now partly fixed in 'fix_tussentaal'
        """
        with doc.retokenize() as retokenizer:
            i = 0
            while i < len(doc) - 1:
                token = doc[i]
                next_token = doc[i + 1]

                # Match: "'s" followed by a lowercase word (e.g. 's morgens, 's avonds)
                if token.text in {"â€˜s","â€™s"} and next_token.is_lower:
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
        """
        fix tussentaal tags: specific tussentaal tags (gij, ge, nen) should have the correct tag (e.g., gij = VNW)
        """
        for token in doc:
            if token.text.lower() in {"gij", "ge"}:  # was originally tagged as noun
                token.tag_ = "VNW|pers"  # Custom tag
                token.pos_ = "PRON"  # Custom POS (pronoun)
            if token.text.lower() in {"nen", "ne", "ene"}:  # was originally tagged as noun
                token.tag_ = "LID|onbep|stan|agr"
                token.pos_ = "DET"
            if token.text.lower() in {"morgens", "avonds","ochtends", "middags", "eens", "waarschijnlijk"}:
                token.tag_ = "BW"
                token.pos_ = "ADV"
            if token.text.lower() =="da":
                token.tag_ = "VNW"
                token.pos_ = "PRON"
            if token.text.lower() == 'beetje':
                token.tag_ = "BW"
                token.pos_ = "ADV"
            if token.text.lower() in {"s", "n"}:
                token.tag_ = "LET"

        return doc

    def fix_nlp_pipeline(self):
        """
        Add aforementioned methods as pipeline components of the Spacy NLP pipeline
        """
        # Only add if not already in pipeline
        if "fix_tussentaal" not in self.nlp.pipe_names:
            self.nlp.add_pipe("fix_tussentaal", last=True)
        if "merge_'s_constructions" not in self.nlp.pipe_names:
            self.nlp.add_pipe("merge_'s_constructions", last=True)


## Call NLP
# epi = epitran.Epitran("nld-Latn")
nlp = spacy.load("nl_core_news_lg")
# Apply my custom fixer
fixer = FixNlpPipeline(nlp)
fixer.fix_nlp_pipeline()




""" CLEAN TRANSCRIPTS FOR SPECIFIC PURPOSES CLASS"""

class CleanTranscript:
    """
    Clean_Transcript Class for cleaning transcripts for different NLP purposes.
    object: a transcript
    """
    def __init__(self, transcript):
        self.transcript = transcript  # this class is namely designed to act on a transcript, so the transcript must
        # be stored inside the class.

    def clean_transcript_for_token_counting(self):
        """
        Clean transcript so that tokens can be counted correctly.

        :return: a transcript where tokens can be counted, including non-words, phonemic language errors, repetitions,
        minimal responses, comments and stereotypes, in accordance with Boxum et al. (2013) and Vandenborre et al. (2018).

        Note:
        - leave in transcript:
           § *p (particles), *a (afgebroken woord), *s and *f (paraphasias), *x and *n (unintelligible words, neologisms)
           § punctuation and spaces: needed for MLU and sentence analysis
        - remove from transcript:
           § elements within brackets (only original utterance should be counted)
           § *g (gevulde pauze), weird annotations/symbols
       - change in transcript:
           § repetitions: consecutive repetitions of a word are reduced to a maximum of two occurrences, with the second
           occurrence annotated as '*r'.
           Aka make sure that a repetition is only counted ONCE. This implies that a repetition is
           defined as 'a word/utterance that has been repeated AT LEAST ONCE'.
           When a word is repeated consecutively multiple times, it will only be counted once. Thus, only one 're-occurrence'
           of that word will be kept in the transcript, all extra (consecutive) re-occurrences are removed.
           The resulting transcript should thus contain repetitions as 1 extra occurrence of the target word, and not
           more than that.
        """
        text = str(self.transcript)

        cleaned_text = re.sub(r'\(.*?\)', '', text)  # Remove any utterance inside brackets ()
        # why: e.g., in semantic paraphasias, now only count paraphasia and not normalized version

        doc = nlp(cleaned_text)  # read transcript into nlp-doc
        cleaned_tokens = []

        # Clean transcript
        for token in doc:
            # if token.is_punct or token.is_space:  # built-in function of token class in Spacy
                # continue
            if '*g'in str(token):  # annotation (gevulde pauze)
                continue
            if 'Ã' in str(token) or '©' in str(token) or 'â' in str(token) or '€' in str(token) or '¦' in str(token):  # rare characters
                continue
            else:
                cleaned_tokens.append(str(token))  # this must be turned into a string to later be able to join all
                # elements again


        # Change repetitions: Remove extra consecutive repetitions, keep max 2
        limited_repetitions_cleaned_tokens = []
        repeated_tokens = []  # just a list to see the repetitions when debugging
        prev_token = None
        repeat_count = 0

        for token in cleaned_tokens:
            if token == prev_token:
                repeated_tokens.append(token)
                repeat_count += 1
                if repeat_count < 2:  # Only allow up to 1 repetition (i.e., total of 2 occurrences) and give it an annotation
                    limited_repetitions_cleaned_tokens.append(''.join([str(token),annot_repetition]))
            else:
                repeat_count = 0
                limited_repetitions_cleaned_tokens.append(token)
                prev_token = token


        cleaned_text = ' '.join(limited_repetitions_cleaned_tokens)
        # why join these again: the *h element is seen as a string after the
        # manipulation. It must however be seen as a token again, and Spacy then requires to read the transcript again
        # into an NLP-readable form (you cannot 'create spacy tokens').

        return cleaned_text



    def clean_transcript_for_tagging(self):
        """
        Clean transcript so that tokens can be tagged correctly.

        :return: a transcript where tokens can be tagged

        Note:
        - leave in transcript:
            § elements within brackets (only normalized utterance should be tagged, original utterance cannot be tagged)
        - remove from transcript:
            § punctuation, *g (gevulde pauze), weird annotations/symbols
            § *p (particles), *a (afgebroken woord), *s and *f (paraphasias), *x and *n (unintelligible words, neologisms)
            § 'yyy' (unintelligible normalized words)
            § Parts within 'hernemingen' that are 'untaggable': only keep the correct part of the herneming (only keep the
             'correct' word 'yyy' from the 'herneming' (xxx-yyy*h) (e.g., ge-geschoten*h)
             § Repetitions: only allow the first occurrence of the target word, otherwise risk that repetition is taken
             into account when tagging (the *r is often misclassified)
        """
        text = str(self.transcript)
        doc = nlp(text)  # read transcript into nlp-doc
        cleaned_tokens = []

        for token in doc:
            if token.is_punct or token.is_space:  # built-in function of token class in Spacy
                continue
            if '*g'in str(token) or '*p' in str(token) or '*a' in str(token):  # annotations
                continue
            if 'Ã' in str(token) or '©' in str(token) or 'â' in str(token) or '€' in str(token) or '¦' in str(token):  # rare characters
                continue
            if '*s' in str(token) or '*S' in str(
                    token):  # make sure to skip the semantic parafasias (it will count only the function of the normalized version)
                continue
            if '*F' in str(token) or '*Fs' in str(token) or '*Fa' in str(token) or '*Fo' in str(token) or '*Ft' in str(token):
                continue  # make sure to skip the phonematic paraphasias (it will count only the function of the normalized version)
            if '*x' in str(token) or '*n' in str(token):
                continue  # make sure to skip unintelligible words and neologisms: you cannot know their function anyways
            if 'yyy' in str(token):
                continue
            if '*h' in str(token):  # only keep the 'correct' word 'yyy' from the 'herneming' (xxx-yyy*h) (e.g., ge-geschoten*h)
                match = re.match(r'(\w+)-(\w+)\*h', token.text)
                if match:
                    cleaned_tokens. append(match.group(2))
            else:
                cleaned_tokens.append(str(token))  # this must be turned into a string to later be able to join all
                # elements again

        # Change repetitions: Remove extra consecutive repetitions, keep max 2
        no_repetitions_cleaned_tokens = []
        repeated_tokens = []  # just a list to see the repetitions when debugging
        prev_token = None
        repeat_count = 0

        for token in cleaned_tokens:
            if token == prev_token:
                repeated_tokens.append(token)
                repeat_count += 1
            else:
                repeat_count = 0
                no_repetitions_cleaned_tokens.append(token)  # allow only the first occurrence of the target word, no repetitions
                prev_token = token

        cleaned_text = ' '.join(no_repetitions_cleaned_tokens)
        # why join these again: the *h element is seen as a string after the
        # manipulation. It must however be seen as a token again, and Spacy then requires to read the transcript again
        # into an NLP-readable form (you cannot 'create spacy tokens').

        return cleaned_text



""" TOKEN/WORD COUNTER CLASS """

class TokenCounter:
    """
    Token_Counter Class
    object: a transcript
    """

    def __init__(self, transcript):
        self.transcript = transcript  # this class is namely designed to act on a transcript, so the transcript must
        # be stored inside the class.

    def total_number_of_tokens(self):
        """

        :return: ALL tokens, including punctuation (excluding whitespace: is the tokenizer splitter)

        Notes:
        - suffixes or prefixes are kept APART (e.g., 's avonds = 's + avonds)
        - abbreviations are kept TOGETHER
        """
        text = str(self.transcript)
        doc = nlp(text)  # read transcript into nlp-doc
        token_count = len(doc)  # (doc is immediately split into tokens, based on whitespaces)

        return token_count

    def total_number_of_words(self):
        """
        Calculate total number of words in the transcript.

        :return: ALL words, including non-words, phonemic language errors, repetitions, minimal responses,
        comments and stereotypes, in accordance with Boxum et al. (2013) and Vandenborre et al. (2018).

        Notes:
        - excludes punctuation!
        TODO: count particles? here yes, volgens Spontaal niet!
        """
        cleaner = CleanTranscript(self.transcript)  # make instance of the class for this text
        cleaned_text = cleaner.clean_transcript_for_token_counting()  # clean the text
        # see helper function to clean transcripts for token counting
        cleaned_text_str = str(cleaned_text)  # make string out of transcript

        doc = nlp(cleaned_text_str)  # read transcript into nlp-doc
        tokens_list = list()

        for token in doc:
            if token.is_punct or token.is_space:  # built-in function of token class in Spacy
                continue
            else:
                tokens_list.append(token)
        nb_words = len(tokens_list)

        return nb_words

    def total_number_of_word_types(self):
        """
        Calculate total number of word types (aka unique words) in the transcript.

        :return: ALL words, including non-words, phonemic language errors, repetitions, minimal responses,
        comments and stereotypes, in accordance with Boxum et al. (2013) and Vandenborre et al. (2018).
        BUT excluding duplicates

        Notes:
        - excludes punctuation!
        """
        cleaner = CleanTranscript(self.transcript)  # make instance of the class for this text
        cleaned_text = cleaner.clean_transcript_for_token_counting()  # clean the text
        # see helper function to clean transcripts for token counting
        cleaned_text_str = str(cleaned_text)  # make string out of transcript

        doc = nlp(cleaned_text_str)  # read transcript into nlp-doc
        tokens_list = list()

        for token in doc:
            if token.is_punct or token.is_space:  # built-in function of token class in Spacy
                continue
            if annot_repetition in str(token):  # make sure the repetition is not counted as a unique item because it
                # has been annotated with a '*r' and thus would 'seem' different from the original target word while
                # it is in fact not different.
                continue
            else:
                tokens_list.append(str(token))  # make sure to convert the token to a string, otherwise 'set' function
                # can't read it
        nb_words = len(tokens_list)
        nb_unique_words = len(set(tokens_list))  # calculate total number of types in this transcript,
        # by allowing no duplicates: how many types per token, for all tokens

        return nb_unique_words



""" TAGGER CLASS """

class POSTagger(object):
    """
    POS_Tagger class
    object: a transcript

    # Tags list Spacy: https://spacy.io/models/nl --> see 'label scheme' --> tagger
        # Glossary of Spacy tags: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
        # https://github.com/UniversalDependencies/UD_Dutch-Alpino/blob/master/stats.xml
        # https://github.com/rug-compling/Alpino/blob/master/AlpinoUserGuide.pdf

    """
    def __init__(self, transcript):
        self.transcript = transcript  # this class is namely designed to act on a transcript, so the transcript must
        # be stored inside the class.

    def show_tag_types(self):
        """
        :return: Show all main Spacy tags in a dictionary
        Note:
        - https://spacy.io/models/nl --> go to 'attributes' within the nl_core_news_lg model
        - if you want extra information, use spacy.explain(...tag..)
        """

        tag_dictionary = {
            'ADJ': 'adjectief - adjective',
            'BW': 'bijwoord - adverb',
            'LET': 'letter - letter + PUNCTUATION',
            'LID': 'lidwoord - article',
            'N': 'zelfstandig naamwoord - noun',
            'SPEC': 'speciaal - special',
            'SPEC|afk': 'afkorting - abbreviation',
            'SPEC|deeleigen': '??',
            'SPEC|enof': '??',
            'SPEC|meta': '??',
            'SPEC|symb': 'symbool - symbol',
            'SPEC|vreemd': 'vreemd woord/leenwoord - foreign word + ANNOTATED words (except *h) ',
            'TSW': '??',
            'TW':'telwoord - cardinal/ordinal',
            'TW|hoofd': 'hoofdtelwoord - cardinal',
            'TW|rang' : 'rangtelwoord - ordinal',
            'VG': 'voegwoord - conjunction',
            'VG|neven': 'nevenschikkend voegwoord - coordinating conjunction',
            'VG|onder': 'onderschikkend voegwoord - subordinating conjunction',
            'VNW': 'voornaamwoord - pronoun',
            'VNW|aanw': 'aanwijzend voornaamwoord - demonstrative pronoun (die, dat)',
            'VNW|betr': 'betrekkelijk voornaamwoord - relative pronoun (die, dat)',
            'VNW|bez': 'bezittelijk voornaamwoord - possessive pronoun (mijn, zijn)',
            'VNW|excl': '??',
            'VNW|onbep': '??',
            'VNW|pers': 'persoonlijk voornaamwoord - personal pronoun (ik, hij)',
            'VNW|pr': '??',
            'VNW|recip': '?? voornaamwoord - recip pronoun',
            'VNW|refl': 'wederkerend voornaamwoord - reflexive pronoun (zich)',
            'VNW|vb': '??',
            'VZ': 'voorzetsel - preposition',
            'VZ|fin': '?? prefix?',
            'VZ|init': '?? suffix?',
            'VZ|versm': '??',
            'WW': 'werkwoord - verb',
            'WW|inf': 'werkwoord infinitief - verb infinitive',
            'WW | od': 'werkwoord onvoltooid deelwoord - present participle',
            'WW | pv': 'werkwoord persoonsvorm - verb conjugation',
            'WW | vd': 'werkwoord voltooid deelwoord - past participle',
             }
        print(tag_dictionary)
        return tag_dictionary


    def tag_list(self, tag_type):
        cleaner = CleanTranscript(self.transcript)  # make instance of the class for this text
        cleaned_text = cleaner.clean_transcript_for_tagging()  # clean the text
        # see helper function to clean transcripts for tagging
        cleaned_text_str = str(cleaned_text)  # make string out of transcript

        doc = nlp(cleaned_text_str)  # read transcript into nlp-doc
        tag_list = []

        for token in doc:  # append all tokens with the specific tag (tag_type) to a list
            tagged_token = token.tag_  # tagged_token = 'tag_type|other information on tage_type...
            if tagged_token.startswith(tag_type): # note: tagged_token [0] == tag_type is too strict, only looks at
                # first LETTER, not first item
                tag_list.append(token)

        return tag_list


    def tag_count(self, tag_type):
        tagger = POSTagger(self.transcript)  # initiate an instance of the class POStagger for this text
        tag_list = tagger.tag_list(tag_type)  # perform the tag_list function on this instance
        tag_count = len(tag_list)

        return tag_count


    def tag_rate(self, tag_type):
        tagger = POSTagger(self.transcript) # initiate an instance of the class POStagger for this text
        counter = TokenCounter(self.transcript)  # initiate an instance of the class TokenCounter for this text

        tag_count = tagger.tag_count(tag_type) # perform the tag_list function on this instance
        tag_rate = tag_count / counter.total_number_of_words()

        return tag_rate


""" UTTERANCE CLASS """
""" helper functions for utterance class """
def is_enumeration(token
):
    """Check if 'and' is used in an enumeration or list."""

    if token.dep_ == 'cc' and token.text.lower() == 'and':
        conj = [t for t in token.head.conjuncts if t != token.head]
        if token.head.pos_ in {'NOUN', 'ADJ'} and conj:
            return True
        if token.head.pos_ == 'VERB':
            return False
    return False

def is_functional_conjunction(token):
    """Check if conjunction is functional (i.e., not a discourse marker)."""
    if token.dep_ == 'cc':
        return True
    if token.pos_ == 'CCONJ' and token.head.pos_ == 'VERB':
        return True
    return False


def extract_embedded_utterances(text):
    """Detects embedded discourse expressions like 'I don't know', 'you know'.
    For Dutch: "je weet wel|ge weet wel|snap je|snapt ge|ik bedoel"

    Return: List of utterances in the transcript, with embedded utterances 'disembedded' (e.g., instead of A-B1-A-B2, it
    gives A-B)
    """
    pattern = r'(,?\s*(je weet wel|ge weet wel|snap je|snapt ge|ik bedoel)[,\.]?\s*)'
    # builds a regex pattern to match common interjected phrases
    splits = re.split(pattern, text)
    # splits the text wherever this pattern occurs, preserving the interjection as a separate element.
    utterances = []

    buffer = ""
    for seg in splits:
        if seg.strip().lower() in {"je weet wel", "ge weet wel", "snap je", "snapt ge", "ik bedoel"}:
            if buffer.strip():  # returns true if buffer isn't empty
                # Only add the buffer as an utterance if it's not empty after trimming
                utterances.append(buffer.strip())  # flush the pre-interjection part
                buffer = ""  # reset the buffer
            utterances.append(seg.strip()) # add interjection itself
        else:
            buffer += seg
    if buffer.strip():
        utterances.append(buffer.strip()) # flush remaining text
    return utterances


""" Utterance class itself """
class Utterance(object):
    """
    Utterance class
    object: a transcript

    Split transcripts into utterances complying with linguistic criteria
    1) “And”: New utterance unless it’s part of an enumeration or combined action.
    2) Conjunctions: Only separate utterances if used disfluently or non-functionally.
    ( 3) Direct speech: All direct speech after a colon (e.g., He said: "I don’t know.") is treated as a distinct unit.
    Each sentence within the direct quote counts as a separate utterance.)
    4) Embedded utterances: Interjected comments within a sentence are extracted as separate utterances
    (e.g., parentheticals, commas with discourse phrases like "I don't know", etc.).

    """
    def __init__(self, transcript):
        self.transcript = transcript
        # this class is namely designed to act on a transcript, so the transcript must
        # be stored inside the class.


    # --- Main Utterance Splitter ---
    def split_into_custom_utterances(self):
        cleaned_transcript = CleanTranscript(self.transcript).clean_transcript_for_token_counting()
        # see helper function to clean transcripts for token counting
        cleaned_transcript_str = str(cleaned_transcript)  # make string out of transcript

        utterances = []
        current = []

        # CRIT 4: Embedded utterances: Interjected comments within a sentence are extracted as separate utterances.
        processed = extract_embedded_utterances(cleaned_transcript_str)
        # returns chunks of text 'utterance.utterance.utterance' and embedded text 'je weet wel'.
        # e.g., ['utterance.utterance.utterance.','je weet wel.', 'utterance']

        for chunk in processed:
            doc_chunk = nlp(chunk)
            for sent in doc_chunk.sents:  # built-in function of Spacy to parse text in sentences
                tokens = list(sent)  # turn one sentence into list of tokens
                current = []

                for token in tokens:  # go over each token in the sentence

                    # CRIT 1: “And”: New utterance unless it’s part of an enumeration or combined action.
                    # Handle "and"
                    if token.text.lower() == 'en':
                        if not is_enumeration(token):  # if that 'and' is not part of an enumeration
                            if current: # AND if current is not empty,
                                # then the current line should be added as a separate utterance
                                # (as this 'and' initiates
                                # a new separate utterance (which will be the next 'current')
                                utterances.append(' '.join([t.text for t in current]))
                                current = []  # reset current again so that it is empty
                            continue    # if current was empty, then do nothing yet (nl the token 'and' initiates a new sentence
                                        # and will be later appended to current so that current can be built again)

                    # CRIT 2: Other conjunctions: Only separate utterances if used disfluently or non-functionally.
                    elif token.pos_ == 'CCONJ' and token.text.lower() != 'en':
                        if not is_functional_conjunction(token):  # if the conjunction is not functional
                            if current:  # AND if current is not empty
                                # then the current line should be added as a separate utterance
                                # (as this 'non-functional conjunction' initiates
                                # a new separate utterance (which will be the next 'current')
                                utterances.append(' '.join([t.text for t in current]))
                                current = []
                            continue
                    current.append(token)  # the token will be added to the current utterance

                    # EXTRA: If sentence ends, the last utterance should be added to the list of utterances
                    if token.is_sent_end:
                        if current:
                            utterances.append(' '.join([t.text for t in current]))
                            current = []

                if current:  # if the current sentence did not have an ending token, it should still be counted
                    # as a utterance and thus appended to the list of utterances
                    utterances.append(' '.join([t.text for t in current]))
                    current = []

        [utt.strip() for utt in utterances if utt.strip()]  # if the utterance is not empty, return a list of utterances
        return utterances



    def total_number_of_words(self):
        """
        Calculate total number of words in the transcript.

        :return: ALL words, including non-words, phonemic language errors, repetitions, minimal responses,
        comments and stereotypes, in accordance with Boxum et al. (2013) and Vandenborre et al. (2018).

        Notes:
        - excludes punctuation!
        TODO: count particles? here yes, volgens Spontaal niet!
        """
        cleaner = CleanTranscript(self.transcript)  # make instance of the class for this text
        cleaned_text = cleaner.clean_transcript_for_token_counting()  # clean the text
        # see helper function to clean transcripts for token counting
        cleaned_text_str = str(cleaned_text)  # make string out of transcript

        doc = nlp(cleaned_text_str)  # read transcript into nlp-doc
        tokens_list = list()

        for token in doc:
            if token.is_punct or token.is_space:  # built-in function of token class in Spacy
                continue
            else:
                tokens_list.append(token)
        nb_words = len(tokens_list)

        return nb_words



for text in text_list:
    # tagger = POSTagger(text)
    # tagger.tag_list('VZ')
    # tagger.tag_count('N')
    # tagger.tag_rate('N')
    # CleanTranscript(text).clean_transcript_for_token_counting()
    Utterance(text).split_into_custom_utterances()
