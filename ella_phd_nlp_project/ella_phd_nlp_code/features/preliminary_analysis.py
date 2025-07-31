"""Perform a preliminary analysis on the files in the database.
Based on scripts from masterthesisEllaLAW

# - on the text files
# - on the audio files

# which actions are performed:
# - get a list of the paths to all files
# - for the text files: read the transcripts in these files; (for the audio files: read the sounds in these files)
# - get a list of the subject (= participant) and question names, one pair (subject, question) for each file.
"""

import glob
import os
import parselmouth # (is for audio part)
import numpy as np

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    TEXT_DIR, AUDIO_PATIENTU_DIR)
    # TEXT_DIR_DUMMY, AUDIO_PATIENTU_DIR_DUMMY)




def all_paths(file_path: str):

    """Get a list of the paths to the text/audio files, based on the correct directory.

    :param file_path: the text or audio directory (TEXT_DIR (or AUDIO_DIR))
    :return:a list of the paths to the separate text/audio files (with each path as 1 specific task of 1 subject in the list)

    Note: this function is based on the following data organization structure:
        'transcripts' -> immediately all transcripts (not subdivided in subject specific folders)

    """
    all_paths_list = (
            glob.glob(file_path + f"{os.path.sep}sub-a[0-9]*")  # txt files from aphasia patients
            #  the 'sub-a*' looks for all files starting with sub-a
            #  [0-9] means 'any digit and doesn't matter how many digits;
            #  * means 'doesn't matter what comes after this'
            + glob.glob(file_path + f"{os.path.sep}sub-b[0-9]*")  # txt files from control patients (comm partner)
            + glob.glob(file_path + f"{os.path.sep}sub-c[0-9]*")) # txt files from control patients (non-related)

    return all_paths_list


def read_transcripts(
    file_path: str,
):
    """
    Get a list of the transcript contents, with each element representing one transcript of one task of one subject.
    aka following the paths in the all_paths function

    :param file_path: the text directory where all text files are located
    :return:a list of the transcripts (format: list(sub1-Q1, sub1-Q2, sub1-Q3 etc)

    Note: this function is based on the following data organization structure:
        'transcripts' -> immediately all transcripts (not subdivided in subject specific folders)

    """
    transcripts_list = []
    all_paths_text = all_paths(file_path)  # each element of this list is a path to a text file
    # (one specific task of 1 specific subject)
    for i in all_paths_text:
        join_path = os.path.join(file_path, i)  # make a path of the text_dir and each participant's specific txt file
        f = open(join_path)  # open this text to be able to read it
        transcript = f.readlines()  # make a transcript, but will be in form of a list* consisting of one (long) string ["text"]
        transcripts_list.append(transcript[0])  # append the transcript string (only the content, not the one-element list* itself)
        # to the list of transcripts

    return transcripts_list



def read_sounds(
    file_path: str,
):
    """Get a list of the sounds correlating to audio-files, with each element representing one sound of one task of one subject.

    :param file_path: the directory where all the audio_patientonly files are located (so those that already underwent
    diarization and concatenation of only the speech of the patient)
    :return:a list of sounds, with each sound correlating to an audio file
    Note: these are Parselmouth sound objects

    Note: this function inputs 1 narrative task (= a merge of the two story tasks: stroke + weekend), so do not input story_stroke
    and story_weekend separately
    """
    sounds_list = []
    all_paths_audio = all_paths(file_path)
    for i in all_paths_audio:
        join_path = os.path.join(file_path, i)  # make a path of the audio_dir and each participant's folder
        sound = parselmouth.Sound(join_path)
        # from https://github.com/drfeinberg/PraatScripts/blob/
        # master/Measure%20Pitch%2C%20HNR%2C%20Jitter%2C%20Shimmer%2C%20and%20Formants.ipynb
        # = how to 'read a sound': sound = parselmouth.Sound(voiceID) # read the sound
        # Result: something with 'Parcelmouth_file_XB003Uetc' for each file
        sounds_list.append(sound)  # append this sound to the sounds list

    return sounds_list



def read_sounds_story(
    file_path: str,
):
    """Get a list of the audio contents, with each element representing one sound of one task of one subject.

    :param file_path: will  be the directory
    :return:a list of sounds (format: list(sub1-Q1, sub1-Q2, sub1-Q3 etc)
    Note: these are Parselmouth sound objects

    Note: this function inputs 2 story tasks (stroke, weekend) and concatenates their content. So not applicable if the
    story_stroke and story_weekend have already been merged beforehand.
    """
    sounds_list = []
    all_paths_audio = all_paths(file_path)
    story_list = []
    for i in all_paths_audio:
        i_splitted = i.split('_')
        if 'story' in i_splitted :
            story_list.append(i)
        else:
            join_path = os.path.join(file_path, i)  # make a path of the audio_dir and each participant's folder
            sound = parselmouth.Sound(join_path)
            # from https://github.com/drfeinberg/PraatScripts/blob/
            # master/Measure%20Pitch%2C%20HNR%2C%20Jitter%2C%20Shimmer%2C%20and%20Formants.ipynb
            # = how to 'read a sound': sound = parselmouth.Sound(voiceID) # read the sound
            # Result: something with 'Parcelmouth_file_XB003Uetc' for each file
            sounds_list.append(sound)  # append this sound to the sounds list
    for item in range(0,len(story_list)-1):
        audio_1 = story_list[item]
        audio_2 = story_list[item+1]
        if audio_1.split('_')[-3] == audio_2.split('_')[-3]:
            sound_1 = parselmouth.Sound(os.path.join(file_path, story_list[item]))
            sound_2 = parselmouth.Sound(os.path.join(file_path, story_list[item+1]))
            combined_sound = parselmouth.Sound.concatenate([sound_1, sound_2])
            sounds_list.append(combined_sound)

    return sounds_list


def get_subject_question_names_from_files(
    file_path: str,
):
    """
    Get the names of the subjects and of the questions according to the order of the text or audio file names.

    param: file_path: the text directory where all text (or audio) files are located
    Note: this should be the same file_path as the all_paths function
    return: a list of the subject and of the question names, according to the order of the text or audio file names
    """
    list_of_subject_question_names = []
    all_paths_list = all_paths(file_path)  # paths to each separate file
    for i in all_paths_list:  # i is one specific file in the all_paths_list
        join_path = os.path.join(file_path, i)  # make a path of the dir (text/audio) and each file
        # example result: C:\\Users\\Ella\\git\\masterthesisEllaLAW\\data\\raw\\transcripts\\s08_actua_BL.txt
        # note: could be 'dubbelop'
        file_name = os.path.basename(join_path)
        # get the filename from the join path.
        # From: https://favtutor.com/blogs/get-filename-from-path-python
        splitted_file_name = file_name.split("_")
        subject_name = splitted_file_name[0]
        question_name = splitted_file_name[2].split(".")[0]
        # [2] because [1] is the transcription (in the text files) or the 'patientonly' (in the audio files)
        # without the split: would be TASK_NAME.txt; with the split: TASK_NAME (idem for audio files: .wav)
        list_of_subject_question_names.append([subject_name, question_name])

    return list_of_subject_question_names


if __name__ == "__main__":
    # all_paths(TEXT_DIR_DUMMY)
    # read_transcripts(TEXT_DIR_DUMMY)
    # get_subject_question_names_from_files(TEXT_DIR_DUMMY)

    # all_paths(AUDIO_PATIENTU_DIR)
    read_sounds(AUDIO_PATIENTU_DIR)