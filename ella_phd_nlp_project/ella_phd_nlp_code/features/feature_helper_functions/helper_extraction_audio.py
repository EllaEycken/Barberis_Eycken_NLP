""" Helper functions to extract linguistic features from audio-files """

""" IMPORT STATEMENTS """

import statistics
import math
from parselmouth.praat import call  # if installing parselmouth: pip install praat-parselmouth

import epitran
import nltk
import pandas as pd
import spacy
import io

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    AUDIO_PATIENT_DIR_DUMMY, # TODO: if all is finished, switch this to AUDIO_PATIENT_DIR!
    f0min,
    f0max,
    period_ceiling,
    unit,
    minpause,
    mindip,
    silencedB,
)

from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import read_sounds

# TODO: add constants, other functions if necessary!
# TODO: change minpause constant (see NLP DOCX)!!!


def calculate_duration(sound: object):
    """Calculate the duration of a sound in PRAAT via Parselmouth (based on Dr Feinberg).

    :param sound
    :return: a float: the duration of the sound

    """
    duration = call(sound, "Get total duration")
    return duration


def create_IntensityObject(
    sound: object,
):
    """Calculate an Intensity object from PRAAT via Parselmouth (based on Dr Feinberg).

    :param sound: object
    :return: an Intensity Object
    create intensity object from the sound file
    DEF: An Intensity object represents an intensity contour at linearly spaced time points ti = t1 + (i – 1) dt,
    with values in dB SPL, i.e. dB relative to 2·10-5 Pascal, which is the normative auditory threshold
    for a 1000-Hz sine wave
    from: https://www.fon.hum.uva.nl/praat/manual/Intensity.html

    Equivalent steps in PRAAT: open sound (is then PRAAT object) --> To Intensity...
    Window in PRAAT with same questions:
    - Minimum pitch (Hz): 100.0 (standard in PRAAT)
    - time step (s): 0.0 (= auto)

    """
    intensityObject = call(sound, "To Intensity", 100, 0.0)
    return intensityObject


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


def create_textGridDataframe(textGrid: object):
    r"""Calculate a TextGrid Dataframe from PRAAT via Parselmouth (based on Dr Feinberg).

    :param textGrid: can be Text Grid Silences Object or Text Grid vuv Object
    :return: a Python dataframe, derived from a PRAAT TextGrid Object
    A PRAAT TextGrid  Object can be 'tabulated' in PRAAT to a PRAAT table, but this table isn't 'readable' in
    Python. To make it readable and to be able to manipulate the values in the table, it must be converted to
    a Python dataframe.

    First step: 'tabulate' the PRAAT TextGrid to a PRAAT Table:
    make a table in which the zeroth column = row, first column = tmin, second column = tier name
    ('what do we look for': silences or vuv), 3rd column = annotation: text: option A vs option B; fourth column = tmax
    Equivalent steps in PRAAT: click on textGrid --> Tabulate --> Down to table
    Window in PRAAT with same questions:
    - Include line number: "no' (in fact cross off)
    - Time decimals: 6
    - Include tier names: "yes"
    - Include empty intervals: "no"

    Second step: convert the PRAAT Table to a Python Pandas Dataframe:
    from: https://groups.google.com/g/parselmouth/c/J2HNzMD5v64
    there mentioned as: As a Plan B, I can use the `pd.read_csv(io.StringIO(parselmouth.praat.call
    (data_table, "List", True)), sep='\t')` trick.

    """
    table_textGrid = call(textGrid, "Down to Table", "no", 6, "yes", "no")
    df_textGrid = pd.read_csv(io.StringIO(call(table_textGrid, "List", True)), sep="\t")
    return df_textGrid


def calculate_syllNucleiFeinberg(
    file_path: str,
):
    """Calculate the nuclei of syllables using PRAAT via Parselmouth (based on Dr Feinberg).

    :param file_path: string
    :return: list of floats containing the 1) Speaking Rate, 2) Articulation rate and 3) average duration of syllables
    with
    1) Speaking Rate = number of syllables / total duration of the signal
    2) Articulation Rate = number of syllables/ phonation or 'sounding' duration (without silences in the original sound)
    3) Pause Rate = number of pauses/total duration of the signal
    4) Average Duration of Syllables = total duration of the phonation 'sounding' duration / number of syllables
    (definitions from sources here underneath)
    in the audio files, using PRAAT via Parselmouth (based on and adapted from Dr Feinberg)
    https://github.com/drfeinberg/PraatScripts/blob/master/syllable_nuclei.py
    = Praat Script Syllable Nuclei: Copyright (C) 2008  Nivja de Jong and Ton Wempe
    Based on paper:
    https://www.researchgate.net/publication/
    24274554_Praat_script_to_detect_syllable_nuclei_and_measure_speech_rate_automatically

    DEF: syllable  duration  is  an  alternative  representation  of articulation  rates.
    from: https://www.researchgate.net/publication/
    327389620_Acoustic_Analysis_of_Whispery_Voice_Disguise_in_Mandarin_Chinese#pf2

    """
    avDurSyll_list = []  # define a now still empty list of average syllable durations of the audio-files
    speakingRate_list = []  # define a now still empty list of speaking rates across the audio files
    articulationRate_list = []  # idem for articulation rate
    pauseRate_list = []  # idem for pause rate
    list_of_sounds = read_sounds(file_path)  # make a list of audio_files with the read-function
    for sound in list_of_sounds:  # for each item in this list of audio files
        duration = calculate_duration(sound)  # calculate the original duration of the sound

        # Step 1: Extract the intensity Object and measure the 'almost max intensity'
        intensityObject = create_IntensityObject(sound)
        # create an intensity object from the sound
        # NOTE: the authors choose 50 Hz as minimum pitch, while we set it to 100 Hz.
        min_intensity = call(intensityObject, "Get minimum", 0, 0, "Parabolic")
        # get minimum intensity of the sound, needed later
        # Equivalent steps in PRAAT: select intensityObject --> query --> get minimum
        # Window in PRAAT:
        # - time range: 0.0   0.0 (= all)
        # - Interpolation: parabolic
        max_intensity = call(intensityObject, "Get maximum", 0, 0, "Parabolic")
        # idem as minimum, needed later
        max_99_intensity = call(intensityObject, "Get quantile", 0, 0, 0.99)
        # What: get the .99 quantile to get the 'almost maximum':
        # Why: Almost maximum (.99 quantile) is used rather than maximum to avoid
        # using irrelevant non-speech sound-bursts.
        # NOTE: in the original paper, then mention to use the median intensity and not the .99 max; but in the script
        # (dr Feinberg, same authors) they mention to change this to .99 max, so that later the treshold can be
        # measured as .99 max - 25 dB (with 25 db being in line with the standard setting to detect silence in the
        # 'To Textgrid (silences)' function (see also masterthesisellalaw.constants: silencedB)
        # Equivalent steps in PRAAT: select intensityObject --> query --> get quantile..
        # Window in PRAAT:
        # - time range: 0.0   0.0 (= all)
        # - Quantile (0-1): 0.99
        # result: X dB

        #  Step 2: Estimate the intensity threshold, as all peaks above it are considered potential syllables
        syll_threshold = max_99_intensity + silencedB
        # what: threshold = .99 max - 25 db (and silencedB = -25 dB so: +(-25 dB) = -25 dB)
        if syll_threshold < min_intensity:
            syll_threshold = min_intensity
            # if the syllable treshold is lower than the minimum intensity of the sound, then set the syll treshold
            # to be this minimum intensity (otherwise you would find no syllables at all)
        nssbursts_threshold = max_intensity - max_99_intensity
        # measure this threshold to see whether there is a big difference between the max intensity and the 'almost'
        # max intensity: if so, there are a lot of non-speech sound bursts that we do not want to look at
        # BUT at the same time that could mean that there is still a lot of 'possible speech sound' in the
        # 0.01 quantile of the intensityObject => so  at the opposite side of the intensity object,
        # (nl at the most silent side), there is a margin to be taken into account (see then treshold 3)
        silence_threshold = silencedB - nssbursts_threshold
        # threshold 2 is the difference between the max int. and the .99 int, and so is the extra margin that has to be
        # taken into account when looking at the most 'silent' parts of the sound: the original silence treshold is
        # LOWERED to account for the 0.01 quantile.

        # Step 2a: get the pauses (silences) to calculate the total speaking time to later calculate the
        # average syllable duration as (total speaking time/ amount of syllables)
        textGridSilences = create_textGridSilencesObject(sound, silenceTreshold=silence_threshold)
        # create a textgrid of the silences, but change the default silenceTreshold parameter from silencedB to
        # silence_treshold calculated above (to include all intensities, see above)
        df_silences_af = create_textGridDataframe(textGridSilences)
        nb_of_pauses_af = 0
        total_sounding_duration = 0
        for i in range(0, len(df_silences_af)):  # go over rows of silences table
            if df_silences_af.iloc[i]["text"] == "sounding":
                # if the cell in row i and column 'text' == 'sounding':
                tstart = df_silences_af.iloc[i]["tmin"]
                tstop = df_silences_af.iloc[i]["tmax"]
                # https://stackoverflow.com/questions/16729574/how-can-i-get-a-value-from-a-cell-of-a-dataframe
                deltat = tstop - tstart
                # the duration of this 'sounding interval or segments' is the start of the sounding interval subtracted
                # from the stop of the sounding interval
                total_sounding_duration += deltat
                # increase the value of the total duration of the sounding part of the audio file with this duration of
                # one of the sounding intervals in the audio file
            else:
                nb_of_pauses_af += 1
                # increase the value of the number of pauses in this audio file with 1 as the column 'silences' has
                # the label 'silent' in this row i and is thus a silent pause interval.
        # NOTE: in the original script they do this in another way (without the use of dataframes) but is same result

        # Step 3: create a new sound that has information on the peaks in the signal
        intensityMatrix = call(intensityObject, "Down to Matrix")
        # why: from this matrix, you can 'convert' the matrix back into a sound file. That sound file only has
        # intensity information, all pitch information is disregarded: in this way: you can easily measure int peaks.
        new_sound = call(intensityMatrix, "To Sound (slice)", 1)
        # create the new sound that is based on intensity information
        # Equivalent steps in PRAAT: select matrix --> synthesise: cast: to sound (slice)
        # Window: row number: 1 (this matrix also only has 1 row, nl intensity values)
        call(new_sound, "Get total duration")
        # use total duration, not end time, to find out duration of new sound
        # in order to allow nonzero starting times.
        call(new_sound, "Get maximum", 0, 0, "Parabolic")
        # same function as above to calculate maximum
        pointProcessObject = call(new_sound, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
        # create a pointProcess object from the sound, that now is focused on points that are not periodic (so not
        # Pointprocess (periodic, cc), but that align with the extrema in the sound.
        # Equivalent steps in PRAAT: select new_sound --> analyse Periodicity: To Pointprocess (extrema):
        # Window in PRAAT:
        # - channel (number, left or right): Left (is just searching technique from left or right edge of the interval)
        # - include maxima: yes
        # - include minima: no
        # interpolation: sinc70
        numpeaks = call(pointProcessObject, "Get number of points")
        # get the number of peaks based on the points in the pointprocess Object
        t = [call(pointProcessObject, "Get time from index", i + 1) for i in range(numpeaks)]
        # estimate time points of the peak positions (of all peaks)

        # Step 3b: In this new sound, inspect the preceding dip in intensity and consider only as a potential syllable
        # 1) a peak that has an intensity above the the syllable treshold
        # AND 2) a peak with a preceding dip of at least 2 or 4 dB with respect to the current peak
        # (note: 2 dB relative dip for unfiltered sound; 4 db for filtered sound: here UN filtered)
        # Step 3b1: fill array with intensity values for all peaks that have an intensity above the syllable treshold
        # aka condition 1
        timepeaks = []
        peakcount = 0
        intensities = []
        for i in range(numpeaks):  # go over all peaks in the signal
            value = call(new_sound, "Get value at time", t[i], "Cubic")  # calculate the intensity value at that peak
            if value > syll_threshold:  # if that intensity is above the syll_treshold
                peakcount += 1  # then the first condition is met
                intensities.append(value)
                timepeaks.append(t[i])
        # Step 3b2: fill array with valid peaks: only intensity values if preceding dip in intensity
        # is greater than mindip, aka condition 2
        validpeakcount = 0
        currenttime = timepeaks[0]
        currentint = intensities[0]
        validtime = []
        for p in range(peakcount - 1):  # go over all peaks that met the first condition
            following = p + 1  # set the following peak as p + 1
            followingtime = timepeaks[p + 1]  # get the time of that following peak
            dip = call(intensityObject, "Get minimum", currenttime, followingtime, "None")
            # calculate the dip in intensity by getting the minimum value between the current peak time and the
            # following peak time
            diffintensity = abs(currentint - dip)  # calculate the difference in intensity by subtracting the intensity
            # of the dip from the intensity of the current peak
            if diffintensity > mindip:  # if that intensity difference (aka 'distance in intensity between this peak
                # and the following, is greater than the minimal dip
                validpeakcount += 1  # then the condition is met and it is a potential syllable
                validtime.append(timepeaks[p])
            currenttime = timepeaks[following]  # set the following  time to be the new current time to make sure that
            # the loop keeps running
            currentint = call(intensityObject, "Get value at time", timepeaks[following], "Cubic")  # idem for intensity

        # Step 4: Step 4. We  extract the pitch  contour, this  time using a window size of 100 msec
        # and 20-msec time steps, and exclude all peaks that are unvoiced as unvoiced peaks can - by definition of
        # a syllable- not be a syllable nuclei (syllable nuclei are vowels and thus have to be voiced)
        pitchObject = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)

        voicedcount = 0
        voicedpeak = []
        for index in range(validpeakcount):  # go over 'indices' of the validpeakcount: just count over all valid peaks
            querytime = validtime[index]  # get the valid time point at this index and set it to the time point you want
            # to look into
            whichinterval = call(textGridSilences, "Get interval at time", 1, querytime)
            # get the silent-sounding interval that comprises that time point
            whichlabel = call(textGridSilences, "Get label of interval", 1, whichinterval)
            # check whether that interval has the label 'sounding' or 'silent'
            value = pitchObject.get_value_at_time(querytime)
            # get the value (frequency) in the pitch object that corresponds to the query time
            if not math.isnan(value):  # if this frequency is not 'nan' aka is periodic and thus VOICED:
                if whichlabel == "sounding":  # and if this 'voiced' part has a sounding label and thus is not a
                    # non-speech burst of sound:
                    voicedcount += 1  # then it is indeed a syllable nucleus: voiced, peak, sounding!
                    voicedpeak.append(validtime[index])  # append the time point of this peak (at the index) to the
                    # time points of the valid and voiced peaks

        # Step 6: return the results
        speakingRate = voicedcount / duration
        articulationRate = voicedcount / total_sounding_duration
        pauseRate = nb_of_pauses_af / duration
        avDurSyll = total_sounding_duration / voicedcount
        speakingRate_list.append(speakingRate)
        articulationRate_list.append(articulationRate)
        pauseRate_list.append(pauseRate)
        avDurSyll_list.append(avDurSyll)

    return speakingRate_list, articulationRate_list, pauseRate_list, avDurSyll_list
