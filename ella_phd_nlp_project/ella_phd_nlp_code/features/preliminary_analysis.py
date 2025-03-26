"""Perform a preliminary analysis on the files in the database.
Based on scripts from masterthesisEllaLAW

# - on the text files
# - on the audio files --> # TODO: not necessary yet? (for NLP project)

# which actions are performed:
# - get a list of the paths to all files
# - for the text files: read the transcripts in these files; for the audio files: read the sounds in these files
# - get a list of the subject (= participant) and question names, one pair (subject, question) for each file.
"""

import glob
import os

# import parselmouth (is for audio part)

from ella_phd_nlp_project.ella_phd_nlp_code.constants import TEXT_DIR_DUMMY # TODO: change to TEXT_DIR if everything is in order!




def all_paths(file_path: str):
    """Get a list of the paths to the text/audio/audio shortened files, based on the correct directory.

    :param file_path: will now be the directory (TEXT_DIR (or AUDIO_DIR))
    :return:a list of the paths to the files (with each path as 1 specific task part of 1 subject in the list)

    Note: how to create all_paths: with REGULAR EXPRESSIONS:
    Alternative options
    1) with a slash: glob.glob(TEXT_DIR + r"*") (= the 'windows version')
    with the slash: look out for anything (on level of subdirectories: all folders underneath TEXT_DIR
    2) glob.glob(TEXT_DIR + f"{os.path.sep}s*")
    the 's*' looks for all files starting with s (here same as everything starts with s)
    3) glob.glob(TEXT_DIR + f"{os.path.sep}s[0-9]*")
    [0-9] means 'any digit and doesn't matter how many digits; * means 'doesn't matter what comes after this'
    => last one is our preferred method

    """
    all_paths_list = (
            glob.glob(file_path + f"{os.path.sep}sub-a[0-9]*")  # docx from aphasia patients
            #  the 'sub-a*' looks for all files starting with sub-a
            #  [0-9] means 'any digit and doesn't matter how many digits;
            #  * means 'doesn't matter what comes after this'
            + glob.glob(file_path + f"{os.path.sep}sub-b[0-9]*")  # docx from control patients (comm partner)
            + glob.glob(file_path + f"{os.path.sep}sub-c[0-9]*")) # docx from control patients (non-related)

    return all_paths_list


def read_transcripts(
    file_path: str,
):
    """Get a list of the transcripts.

    :param file_path: will now be the directory, NOT the join_path as you make a path during this function
    :return:a list of the transcripts (format: list(P1Q1, P1Q2, P1Q3 etc)
    function based on new structure: 'transcripts' -> immediately all transcripts (not subdivided in s-folders)

    """
    transcripts_list = []
    all_paths_text = all_paths(file_path)  # each element of this list is a path to a text file
    # (one spec interview of 1 specific subject)
    for i in all_paths_text:
        join_path = os.path.join(file_path, i)  # *make a path of the text_dir and each participant's specific
        f = open(join_path)  # open this text to be able to read it
        transcript = f.readlines()  # make a transcript, but will be in form of a list of a string ["text"]
        transcripts_list.append(transcript[0])
    return transcripts_list


def read_sounds(
    file_path: str,
):
    """Get a list of the sounds based correlating to audio-files.

    :param file_path: will now be the directory, NOT the join_path as you make a path during this function
    :return:a list of sounds, with each sound correlating to an audio file
    Note: these are Parselmouth sound objects
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


def get_subject_question_names_from_files(
    file_path: str,
):
    """Get the names of the subjects and of the questions according to the order of the text or audio file names."""
    list_of_subject_question_names = []
    all_paths_list = all_paths(file_path)  # paths to each separate file
    for i in all_paths_list:  # i is one specific file in the all_paths_list
        join_path = os.path.join(file_path, i)  # make a path of the dir (text/audio) and each file
        # result: C:\\Users\\Ella\\git\\masterthesisEllaLAW\\data\\raw\\transcripts\\s08_actua_BL.txt
        # note: could be 'dubbelop'
        file_name = os.path.basename(join_path)
        # get the filename from the join path.
        # From: https://favtutor.com/blogs/get-filename-from-path-python
        splitted_file_name = file_name.split("_")
        if splitted_file_name[0] not in ["ARCK", "MC"]:
            subject_name = splitted_file_name[0]
            if splitted_file_name[1] == "bio":
                question_name = "_".join([splitted_file_name[1], splitted_file_name[2]])
            else:
                question_name = splitted_file_name[1]
            list_of_subject_question_names.append([subject_name, question_name])
        elif splitted_file_name[0] == "ARCK":
            subject_name = "_".join([splitted_file_name[0], splitted_file_name[1], splitted_file_name[2]])
            if splitted_file_name[3] == "bio":
                question_name = "_".join([splitted_file_name[3], splitted_file_name[4]])
            else:
                question_name = splitted_file_name[3]
            list_of_subject_question_names.append([subject_name, question_name])
        elif splitted_file_name[0] == "MC":
            subject_name = "_".join([splitted_file_name[0], splitted_file_name[1]])
            if splitted_file_name[3] == "bio":
                question_name = "_".join([splitted_file_name[2], splitted_file_name[3]])
            else:
                question_name = splitted_file_name[2]
            list_of_subject_question_names.append([subject_name, question_name])

    return list_of_subject_question_names


if __name__ == "__main__":
    get_subject_question_names_from_files(AUDIO_DIR)