""" Helper functions to extract linguistic features from audio-files """

""" IMPORT STATEMENTS """

import statistics
import math
from parselmouth.praat import call  # if installing parselmouth: pip install praat-parselmouth

import epitran
import nltk
import pandas as pd
import spacy

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (  # TODO: if all is finished, switch this to AUDIO_DIR!
    AUDIO_DIR_DUMMY,
    period_ceiling,
    unit,
)

def calculate_duration(sound: object):
    """Calculate the duration of a sound in PRAAT via Parselmouth (based on Dr Feinberg).

    :param sound
    :return: a float: the duration of the sound

    """
    duration = call(sound, "Get total duration")
    return duration


def create_textGridSilencesObject(sound: object, silenceTreshold=silencedB):
    """Calculate a TextGrid (silences) Object from PRAAT via Parselmouth (based on Dr Feinberg).

    :param silenceTreshold: it is default set to be = silencedB; but if specified, it can be set to another value
        from: https://stackoverflow.com/questions/39064970/changing-functions-default-parameters-in-python
    :param sound: object
    :return: TextGrid (silences) Object with silences as tier layer
    from: see examples (note: example in stackoverflow used Intensity object to make textgrid)
    textGrid = a type of object in PRAAT

    DEF: TextGrid in which the silent and sounding intervals of the selected Sound are marked. Shows
    the zeroth column = row, first column = tmin, second column = tier name ('what do we look for': silences),
    3rd column = annotation: text: 'silent' vs 'sounding'; fourth column = tmax
    From: https://www.fon.hum.uva.nl/praat/manual/Sound__To_TextGrid__silences____.html

    Equivalent steps in PRAAT: use 'sound object'--> Annotate--> To textgrid (silences)...
    Window in PRAAT with questions:
    - parameters for the intensity analysis: minimum Pitch 100 Hz; Time step (s): 0.0
    - Silent intervals detection:
        silence treshold (dB): - 25; => 'silencedB' in masterthesisellalaw.constants
        minimum silent interval (s): 0.25,  => 'minpause' in masterthesisellalaw.constants
            NOTE: what is minimum silence interval length to be seen as pause:
            From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8874014/#B30: 0.1 is absolute minimum but even
            some authors suggest 0.18-0.25 as 0.1 is also silent interval for plosives and thus no pause
            Note: script of dr Feinberg uses even 0.3 as absolute minimum
            From: https://github.com/drfeinberg/PraatScripts/blob/master/syllable_nuclei.py
        minimum sounding interval (s): 0.1,
            NOTE: what is minimum speech interval length to be seen as speech and not as 'filled pause':
            From: https://www.mdpi.com/2226-471X/8/1/79: mean length of filled pauses is ~400 ms
            => should I say that 'speech' minimum interval is 400 ms to exclude filled pauses?
        silent interval label: silent, sounding interval label: sounding
    Coding based on:
    https://stackoverflow.com/questions/34770105/praat-script-to-remove-silence-cannot-select-and-remove-objects
    and
    https://github.com/stylerw/styler_praat_scripts/blob/master/extract_silences.praat

    """
    textGridSilences = call(
        sound,
        "To TextGrid (silences)",
        f0min,
        0.0,
        silenceTreshold,
        minpause,
        0.1,
        "silent",
        "sounding",
    )
    return textGridSilences

