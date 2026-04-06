""" Preprocess audio files from IANSA dataset
Note: running again is not possible if the same wav-file must be overwritten. Then delete original wav-file and run again. """

import os
import re
import glob
import parselmouth



import datasets  # install it first: is a package

from ella_phd_nlp_project.ella_phd_nlp_code.constants import (
    # AUDIO_DIR_DUMMY, DIAR_DIR_DUMMY,  NONMERGED_AUDIO_PATIENT_DIR_DUMMY, AUDIO_PATIENT_DIR_DUMMY,
    # NONMERGED_AUDIO_PATIENTU_DIR_DUMMY, AUDIO_PATIENTU_DIR_DUMMY,
    # CLEAN_DIAR_DIR_DUMMY,
    AUDIO_DIR, DIAR_DIR,
    NONMERGED_AUDIO_PATIENTU_DIR, AUDIO_PATIENTU_DIR,
    concatenation_overlap_time)

from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import *
from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_helper_functions.helper_extraction_audio import *
from ella_phd_nlp_project.ella_phd_nlp_code.data.preprocess_diarization import *



## Filter ONE audio file based on the diarization indices
# -------------------------------------------------------

def filter_audio_file_uninterruptedmerged(raw_audio_path_in, diarization_dir, processed_dir, new_spk_code = 0, overrule_spk_code = False):
    """

    :param raw_audio_path_in: the original audio (wav) file that is stored in the raw directory
    :param diarization_dir: the diarization directory path with diarization files
    :param processed_dir: the processed directory where the filtered audio files will be saved
    :param new_spk_code: if to overwrite spk code, this is new code
    :param overrule_spk_code: whether to overwrite spk code if it exists
    :return: a filtered audio file in the processed directory, containing only audio intervals (concatenated) attributed
    to the patient

    Note: in this function, uninterrupted segments from one speaker are MERGED, in the sense that only
    the starting time of the first segment is counted as the start time and the end time of the LAST segment is counted as the end time.
    All inbetween segments from that same speaker are thus NOT interrupted (and later concatenated). In this way,
    pauses/person noise is KEPT in the audio files.
    (note: if diarization files must be cleaned and temporarily stored in the interim_dir, add 'interim_dir' as param here
    """

    ## Choose the name for the filtered audio file
    audio_name= os.path.splitext(os.path.basename(raw_audio_path_in))[0]
    audio_name_subparts = audio_name.split('_')
    # 'os.path.splitext(os.path.basename(audio_path_in))[0]' extracts the name of the audio_path (all without the .wav
    # extension)
    audio_new_name = str()
    if len(audio_name_subparts) > 2:  # for those where 'story' is involved (name is then longer than 2 items: story_weekend or story_stroke
        audio_new_name = "_".join([audio_name_subparts[0], 'patientonlyU', audio_name_subparts[1], audio_name_subparts[2]])
    else:
        audio_new_name = "_".join([audio_name_subparts[0], 'patientonlyU', audio_name_subparts[1]])
    audio_path_out = os.path.join(
        processed_dir, ".".join(
            [audio_new_name,
             'wav']))

    ## Read the original raw sound
    raw_sound = parselmouth.Sound(raw_audio_path_in)

    ## Read diarization info
    original_diarization_file = ".".join([audio_name,'txt'])
    diarization_file_path = os.path.join(diarization_dir, original_diarization_file)
    # NOT NEEDED: cleaned_diarization_file_path = cleanup_diar_txt_file(diar_txt_path_in, interim_dir)
    spk_code = give_patient_spk_code(diarization_file_path)
    # if cleaned_diar needed, then swap diarization_file_path with cleaned_diarization_file_path
    if overrule_spk_code:
        spk_code = new_spk_code
    all_speaker_intervals = []

    # Read the txt file and put all the information as tuples (start, end, spk) in a list
    with open(diarization_file_path, 'r') as f:  # note: use cleaned_diarization_file_path if needed
        for line in f:
            parts = line.strip().split(',')  # Use comma delimiter
            if len(parts) != 3:
                continue  # Skip malformed lines

            start = float(parts[0])
            end = float(parts[1])
            speaker = int(parts[2])
            all_speaker_intervals.append((start, end, speaker))

    # Keep only the patient segments, merging uninterrupted segments of the same speaker
    # Now merge spk_0 segments
    merged_patientonly_intervals = []
    in_patientonly_block = False
    current_start = None
    current_end = None

    for start, end, speaker in all_speaker_intervals:
        if speaker == spk_code:  # and thus spk_code = patient
            if not in_patientonly_block:
                # Start a new block
                current_start = start
                current_end = end
                in_patientonly_block = True
            else:
                # Extend the current block
                current_end = end
        else:
            if in_patientonly_block:
                # End the current block
                merged_patientonly_intervals.append((current_start, current_end))
                in_patientonly_block = False

    # Edge case: if the last block was spk_code (patientonly thus)
    if in_patientonly_block:
        merged_patientonly_intervals.append((current_start, current_end))

    # Print or return the results
    for s, e in merged_patientonly_intervals:
        print(f"{s:.2f}, {e:.2f}")

    # Extract and collect the patientonly_segments
    extracted_segments = []
    for start, end in merged_patientonly_intervals:
        segment = raw_sound.extract_part(from_time=start, to_time=end, preserve_times=False)
        # segment = apply_fade(segment)  # make sure you apply fade in, fade out so that concatenated sound has smooth transitions
        extracted_segments.append(segment)

    # Concatenate all patientonly_segments into one new sound
    if extracted_segments:
        # patientonly_sound = parselmouth.Sound.concatenate(extracted_segments)
        patientonly_sound = parselmouth.Sound.concatenate(extracted_segments, overlap = concatenation_overlap_time)
        # parselmouth source code: static concatenate(sounds: List[parselmouth.Sound],
        # overlap: NonNegative[float] = 0.0)→ parselmouth.Sound
        # TODO: switch concatenation_overlap_time (in constants) if necessary
        # TODO: WATCH OUT: you cannot create the same file twice, should be different name
        # to smoothen the concatenation

        patientonly_sound.save(audio_path_out, 'WAV')
        print(f" Saved filtered audio to: {audio_path_out}")
        return audio_path_out
    else:
        print("No segments found for the target speaker.")
        return None



def filter_audio_file(raw_audio_path_in, diarization_dir, processed_dir, new_spk_code = 0, overrule_spk_code = False):
    """

    :param raw_audio_path_in: the original audio (wav) file that is stored in the raw directory
    :param diarization_dir: the diarization directory path with diarization files
    :param processed_dir: the processed directory where the filtered audio files will be saved
    :param new_spk_code: if to overwrite spk code, this is new code
    :param overrule_spk_code: whether to overwrite spk code if it exists
    :return: a filtered audio file in the processed directory, containing only audio intervals (concatenated) attributed
    to the patient

    Note: in this function, uninterrupted segments from one speaker are NOT MERGED, in the sense that each speaker segment,
    whether or not it is interrupted by the other speaker, is counted separately. This means that different segments of
    the SAME speaker (so uninterrupted segments) are counted and concatenated separately, implying that breaks within the speech
    of the same speaker (therefore different segments), such as pauses or speaker noise, are filtered OUT of the concatenated
    speaker sound. This could yield in losing information on pauses...
    (note: if diarization files must be cleaned and temporarily stored in the interim_dir, add 'interim_dir' as param here
    """

    ## Choose the name for the filtered audio file
    audio_name= os.path.splitext(os.path.basename(raw_audio_path_in))[0]
    audio_name_subparts = audio_name.split('_')
    # 'os.path.splitext(os.path.basename(audio_path_in))[0]' extracts the name of the audio_path (all without the .wav
    # extension)
    audio_new_name = str()
    if len(audio_name_subparts) > 2:  # for those where 'story' is involved (name is then longer than 2 items: story_weekend or story_stroke
        audio_new_name = "_".join([audio_name_subparts[0], 'patientonly', audio_name_subparts[1], audio_name_subparts[2]])
    else:
        audio_new_name = "_".join([audio_name_subparts[0], 'patientonly', audio_name_subparts[1]])
    audio_path_out = os.path.join(
        processed_dir, ".".join(
            [audio_new_name,
             'wav']))

    ## Read the original raw sound
    raw_sound = parselmouth.Sound(raw_audio_path_in)

    ## Read diarization info
    original_diarization_file = ".".join([audio_name,'txt'])
    diarization_file_path = os.path.join(diarization_dir, original_diarization_file)
    # NOT NEEDED: cleaned_diarization_file_path = cleanup_diar_txt_file(diar_txt_path_in, interim_dir)
    spk_code = give_patient_spk_code(diarization_file_path)
    # if cleaned_diar needed, then swap diarization_file_path with cleaned_diarization_file_path
    if overrule_spk_code:
        spk_code = new_spk_code
    patientonly_intervals = []
    with open(diarization_file_path, 'r') as f:  # note: use cleaned_diarization_file_path if needed
        for line in f:
            parts = line.strip().split(',')  # Use comma delimiter
            if len(parts) != 3:
                continue  # Skip malformed lines

            start = float(parts[0])
            end = float(parts[1])
            speaker = int(parts[2])

            if speaker == spk_code:
                patientonly_intervals.append((start, end))

    # Extract and collect the patientonly_segments
    extracted_segments = []
    for start, end in patientonly_intervals:
        segment = raw_sound.extract_part(from_time=start, to_time=end, preserve_times=False)
        # segment = apply_fade(segment)  # make sure you apply fade in, fade out so that concatenated sound has smooth transitions
        extracted_segments.append(segment)

    # Concatenate all patientonly_segments into one new sound
    if extracted_segments:
        # patientonly_sound = parselmouth.Sound.concatenate(extracted_segments)
        patientonly_sound = parselmouth.Sound.concatenate(extracted_segments, overlap = concatenation_overlap_time)
        # parselmouth source code: static concatenate(sounds: List[parselmouth.Sound],
        # overlap: NonNegative[float] = 0.0)→ parselmouth.Sound
        # TODO: switch concatenation_overlap_time (in constants) if necessary
        # TODO: WATCH OUT: you cannot create the same file twice, should be different name
        # to smoothen the concatenation

        patientonly_sound.save(audio_path_out, 'WAV')
        print(f" Saved filtered audio to: {audio_path_out}")
        return audio_path_out
    else:
        print("No segments found for the target speaker.")
        return None




## Preprocess all IANSA audio files (by filtering them)
# ------------------------------------------------------
def preprocess_IANSA_audio_uninterruptedmerged(raw_dir, diarization_dir, interim_dir, processed_dir, overrule_spk_code_list = None):
    """
    Preprocessing function to filter the audio files based on the speaker diarization codes
    :param raw_dir: the raw audio directory containing the original audio (wav) files
    :param diarization_dir: the raw diarization directory path where the (NON-clean) diarization files are saved
    :param interim_dir: the interim audio directory where the filtered, non-merged (for story_stroke and story_weekend)
    audio files are stored
    :param processed_dir: the processed directory path where the filtered audio files will be stored
    :param overrule_spk_code_list: default None.
    If not none, is list of TUPLES, each tuple = (audio-file NAME (e.g., sub-a043_ANTAT), correct speaker-diarization code).
    Only to fill in if the audio file has been attributed the wrong spk-code
    i.e., the patient has spk-code 2, or the patient began speaking in the audio file so that function give_patient_spk_code
    is wrong (is based on assumption that patient doesn't start speaking in the file).

    :return: a list of the preprocessed audio file in the processed directory, containing only the sounds of the speaker
    Note: story_stroke and story_weekend will be MERGED into story
    Note: if diarization files must be cleaned, add param interim_diarization_dir: the interim directory path where
    the cleaned diarization files will be saved
    """

    ## Define the lists that have to be created
    all_audio_patientonly_nonmerged_files_list = list()
    all_audio_patientonly_files_list = list()

    ## Get a list of all raw audio and diarization file paths
    all_audio_files_list = (
            glob.glob(raw_dir + f"{os.path.sep}sub-a[0-9]*")  # audio files from aphasia patients
            #  the 'sub-a*' looks for all files starting with sub-a
            #  [0-9] means 'any digit and doesn't matter how many digits;
            #  * means 'doesn't matter what comes after this'
            + glob.glob(raw_dir + f"{os.path.sep}sub-b[0-9]*")  # docx from control patients (comm partner)
            + glob.glob(raw_dir + f"{os.path.sep}sub-c[0-9]*")  # docx from control patients (non-related)
            + glob.glob(raw_dir + f"{os.path.sep}sub-[0-9]*")  # docx from acute stroke patients
    )

    all_diarization_files_list = (
            glob.glob(diarization_dir + f"{os.path.sep}sub-a[0-9]*")  # audio files from aphasia patients
            #  the 'sub-a*' looks for all files starting with sub-a
            #  [0-9] means 'any digit and doesn't matter how many digits;
            #  * means 'doesn't matter what comes after this'
            + glob.glob(diarization_dir + f"{os.path.sep}sub-b[0-9]*")  # docx from control patients (comm partner)
            + glob.glob(diarization_dir + f"{os.path.sep}sub-c[0-9]*")  # docx from control patients (non-related)
    )
    if len(all_audio_files_list) != len(all_diarization_files_list):
        print("Not all audio files have a diarization file")


    ## Create a list of audio and spk_code files when they have to overrule the give_spk_code_patient function
    audio_file_overruling_list = list()
    spk_code_overruling_list = list()
    if overrule_spk_code_list:
        for name_spk_tuple in overrule_spk_code_list:
            audio_file_overruling = name_spk_tuple[0]
            audio_file_overruling_list.append(audio_file_overruling)
            spk_code_overruling = name_spk_tuple[1]
            spk_code_overruling_list.append(spk_code_overruling)


    ## Generate the patientonly, but non-merged audio file
    for audio_file in all_audio_files_list:
        audio_file_name = os.path.splitext(os.path.basename(audio_file))[0]
        corresponding_diarization_file_path = os.path.join(
            diarization_dir, ".".join([audio_file_name, 'txt']))
        if corresponding_diarization_file_path not in all_diarization_files_list:
            print(f"Not all audio files have a diarization file: {audio_file}. audio_patientonly for this file could not be generated.")
            continue
        else:
            if audio_file_name in audio_file_overruling_list: # if overruling applies
                index_in_overruling_list = audio_file_overruling_list.index(audio_file_name)
                audio_patientonly_file = filter_audio_file_uninterruptedmerged(
                    raw_audio_path_in=audio_file,
                    diarization_dir=diarization_dir,
                    processed_dir=interim_dir,  # here the NONMERGED audio_patientonly files are stored
                    new_spk_code= spk_code_overruling_list[index_in_overruling_list],
                    overrule_spk_code=True,
                    )
                all_audio_patientonly_nonmerged_files_list.append(audio_patientonly_file)

            else:
                audio_patientonly_file = filter_audio_file_uninterruptedmerged(
                    raw_audio_path_in = audio_file,
                    diarization_dir = diarization_dir,
                    processed_dir = interim_dir,  # here the NONMERGED audio_patientonly files are stored
                    )
                all_audio_patientonly_nonmerged_files_list.append(audio_patientonly_file)


    ## Generate the MERGED patientonly audio files (where the story_stroke and story_weekend are merged into story)
    sub_story_list = list()

    # Append the non-story files to the final list of patientonly_audio files in the processed_dir
    for audio_patientonly_file in all_audio_patientonly_nonmerged_files_list:
        audio_name = os.path.splitext(os.path.basename(audio_patientonly_file))[0]

        if 'story' in audio_name:  # do not append them to the processed dir yet, must be merged first
            audio_name_subject_number = audio_name.split("_")[0]
            if audio_name_subject_number in sub_story_list: # only append the audio_names (subject_story) ONCE to the story_list,
                # so that merged stroke _ weekend story only happens ONCE
                continue
            else:
                sub_story_list.append(audio_name_subject_number)

        else:   #  append the file to the processed dir (should not be merged)
            sound = parselmouth.Sound(audio_patientonly_file)
            audio_path_out = os.path.join(
                processed_dir, ".".join([audio_name,'wav']))
            sound.save(audio_path_out, 'WAV')
            all_audio_patientonly_files_list.append(audio_path_out)

    # For the story-files, merge the 'stroke' and 'weekend' subparts if possible before appending them to the list
    for sub in sub_story_list:
        # define and check file paths and names
        audio_story_name = "_".join([sub, "patientonlyU", "narrative"])
        audio_story_path_out = os.path.join(
            processed_dir, ".".join([audio_story_name, 'wav']))

        audio_story_stroke_name = "_".join([sub,"patientonlyU", "story","stroke"])
        audio_story_stroke_file_path = os.path.join(
            interim_dir, ".".join([audio_story_stroke_name, "wav"])
        )
        audio_story_weekend_name = "_".join([sub,"patientonlyU", "story","weekend"])
        audio_story_weekend_file_path = os.path.join(
            interim_dir, ".".join([audio_story_weekend_name, "wav"])
        )
        # if the stroke story is in the interim dir, then merge it with the weekend story if the latter exists,
        # otherwise set the stroke story as the only story
        if audio_story_stroke_file_path in all_audio_patientonly_nonmerged_files_list:
            if audio_story_weekend_file_path in all_audio_patientonly_nonmerged_files_list:
                story_stroke_sound = parselmouth.Sound(audio_story_stroke_file_path)
                story_weekend_sound = parselmouth.Sound(audio_story_weekend_file_path)
                story_sound = parselmouth.Sound.concatenate([story_stroke_sound,story_weekend_sound],
                                                            overlap = concatenation_overlap_time)
                # parselmouth source code: static concatenate(sounds: List[parselmouth.Sound],
                # overlap: NonNegative[float] = 0.0)→ parselmouth.Sound
                # TODO: switch concatenation_overlap_time (in constants) if necessary
                story_sound.save(audio_story_path_out, 'WAV')
                all_audio_patientonly_files_list.append(audio_story_path_out)

            else:
                story_sound = parselmouth.Sound(audio_story_stroke_file_path)
                story_sound.save(audio_story_path_out, 'WAV')
                all_audio_patientonly_files_list.append(audio_story_path_out)
        # if the weekend story exists (and thus not the stroke story), set this one as the only story for that subject
        elif audio_story_weekend_file_path in all_audio_patientonly_nonmerged_files_list:
            story_sound = parselmouth.Sound(audio_story_weekend_file_path)
            story_sound.save(audio_story_path_out, 'WAV')

     # Now perform a double check whether all story-files have been merged (then called 'narrative')
    for audio_file in all_audio_patientonly_files_list:  # sanity check
        if 'story' in audio_file:  # now there shouldn't be any story file left, only 'narratives'
            print(
                f"Some story files haven't been merged yet: {audio_file}. "
                f"Audio_patientonlyU 'story' for this file has been generated but has not yet been merged with 'story_weekend' "
                f"nor has been checked whether this file is the only 'story' part for this subject (i.e., we don't know"
                f"whether story_weekend exists for this subject.")
            continue

    return all_audio_patientonly_files_list



## Preprocess all IANSA audio files (by filtering them)
def preprocess_IANSA_audio(raw_dir, diarization_dir, interim_dir, processed_dir, overrule_spk_code_list = None):
    """
    Preprocessing function to filter the audio files based on the speaker diarization codes
    :param raw_dir: the raw audio directory containing the original audio (wav) files
    :param diarization_dir: the raw diarization directory path where the (NON-clean) diarization files are saved
    :param interim_dir: the interim audio directory where the filtered, non-merged (for story_stroke and story_weekend)
    audio files are stored
    :param processed_dir: the processed directory path where the filtered audio files will be stored
    :param overrule_spk_code_list: default None.
    If not none, is list of TUPLES, each tuple = (audio-file NAME (e.g., sub-a043_ANTAT), correct speaker-diarization code).
    Only to fill in if the audio file has been attributed the wrong spk-code
    i.e., the patient has spk-code 2, or the patient began speaking in the audio file so that function give_patient_spk_code
    is wrong (is based on assumption that patient doesn't start speaking in the file).

    :return: a list of the preprocessed audio file in the processed directory, containing only the sounds of the speaker
    Note: story_stroke and story_weekend will be MERGED into story
    Note: if diarization files must be cleaned, add param interim_diarization_dir: the interim directory path where
    the cleaned diarization files will be saved
    """

    ## Define the lists that have to be created
    all_audio_patientonly_nonmerged_files_list = list()
    all_audio_patientonly_files_list = list()

    ## Get a list of all raw audio and diarization file paths
    all_audio_files_list = (
            glob.glob(raw_dir + f"{os.path.sep}sub-a[0-9]*")  # audio files from aphasia patients
            #  the 'sub-a*' looks for all files starting with sub-a
            #  [0-9] means 'any digit and doesn't matter how many digits;
            #  * means 'doesn't matter what comes after this'
            + glob.glob(raw_dir + f"{os.path.sep}sub-b[0-9]*")  # docx from control patients (comm partner)
            + glob.glob(raw_dir + f"{os.path.sep}sub-c[0-9]*")  # docx from control patients (non-related)
    )

    all_diarization_files_list = (
            glob.glob(diarization_dir + f"{os.path.sep}sub-a[0-9]*")  # audio files from aphasia patients
            #  the 'sub-a*' looks for all files starting with sub-a
            #  [0-9] means 'any digit and doesn't matter how many digits;
            #  * means 'doesn't matter what comes after this'
            + glob.glob(diarization_dir + f"{os.path.sep}sub-b[0-9]*")  # docx from control patients (comm partner)
            + glob.glob(diarization_dir + f"{os.path.sep}sub-c[0-9]*")  # docx from control patients (non-related)
    )
    if len(all_audio_files_list) != len(all_diarization_files_list):
        print("Not all audio files have a diarization file")


    ## Create a list of audio and spk_code files when they have to overrule the give_spk_code_patient function
    audio_file_overruling_list = list()
    spk_code_overruling_list = list()
    if overrule_spk_code_list:
        for name_spk_tuple in overrule_spk_code_list:
            audio_file_overruling = name_spk_tuple[0]
            audio_file_overruling_list.append(audio_file_overruling)
            spk_code_overruling = name_spk_tuple[1]
            spk_code_overruling_list.append(spk_code_overruling)


    ## Generate the patientonly, but non-merged audio file
    for audio_file in all_audio_files_list:
        audio_file_name = os.path.splitext(os.path.basename(audio_file))[0]
        corresponding_diarization_file_path = os.path.join(
            diarization_dir, ".".join([audio_file_name, 'txt']))
        if corresponding_diarization_file_path not in all_diarization_files_list:
            print(f"Not all audio files have a diarization file: {audio_file}. audio_patientonly for this file could not be generated.")
            continue
        else:
            if audio_file_name in audio_file_overruling_list: # if overruling applies
                index_in_overruling_list = audio_file_overruling_list.index(audio_file_name)
                audio_patientonly_file = filter_audio_file(
                    raw_audio_path_in=audio_file,
                    diarization_dir=diarization_dir,
                    processed_dir=interim_dir,  # here the NONMERGED audio_patientonly files are stored
                    new_spk_code= spk_code_overruling_list[index_in_overruling_list],
                    overrule_spk_code=True,
                    )
                all_audio_patientonly_nonmerged_files_list.append(audio_patientonly_file)

            else:
                audio_patientonly_file = filter_audio_file(
                    raw_audio_path_in = audio_file,
                    diarization_dir = diarization_dir,
                    processed_dir = interim_dir,  # here the NONMERGED audio_patientonly files are stored
                    )
                all_audio_patientonly_nonmerged_files_list.append(audio_patientonly_file)


    ## Generate the MERGED patientonly audio files (where the story_stroke and story_weekend are merged into story)
    sub_story_list = list()

    # Append the non-story files to the final list of patientonly_audio files in the processed_dir
    for audio_patientonly_file in all_audio_patientonly_nonmerged_files_list:
        audio_name = os.path.splitext(os.path.basename(audio_patientonly_file))[0]

        if 'story' in audio_name:  # do not append them to the processed dir yet, must be merged first
            audio_name_subject_number = audio_name.split("_")[0]
            if audio_name_subject_number in sub_story_list: # only append the audio_names (subject_story) ONCE to the story_list,
                # so that merged stroke _ weekend story only happens ONCE
                continue
            else:
                sub_story_list.append(audio_name_subject_number)

        else:   #  append the file to the processed dir (should not be merged)
            sound = parselmouth.Sound(audio_patientonly_file)
            audio_path_out = os.path.join(
                processed_dir, ".".join([audio_name,'wav']))
            sound.save(audio_path_out, 'WAV')
            all_audio_patientonly_files_list.append(audio_path_out)

    # For the story-files, merge the 'stroke' and 'weekend' subparts if possible before appending them to the list
    for sub in sub_story_list:
        # define and check file paths and names
        audio_story_name = "_".join([sub, "patientonly", "narrative"])
        audio_story_path_out = os.path.join(
            processed_dir, ".".join([audio_story_name, 'wav']))

        audio_story_stroke_name = "_".join([sub,"patientonly", "story","stroke"])
        audio_story_stroke_file_path = os.path.join(
            interim_dir, ".".join([audio_story_stroke_name, "wav"])
        )
        audio_story_weekend_name = "_".join([sub,"patientonly", "story","weekend"])
        audio_story_weekend_file_path = os.path.join(
            interim_dir, ".".join([audio_story_weekend_name, "wav"])
        )
        # if the stroke story is in the interim dir, then merge it with the weekend story if the latter exists,
        # otherwise set the stroke story as the only story
        if audio_story_stroke_file_path in all_audio_patientonly_nonmerged_files_list:
            if audio_story_weekend_file_path in all_audio_patientonly_nonmerged_files_list:
                story_stroke_sound = parselmouth.Sound(audio_story_stroke_file_path)
                story_weekend_sound = parselmouth.Sound(audio_story_weekend_file_path)
                story_sound = parselmouth.Sound.concatenate([story_stroke_sound,story_weekend_sound],
                                                            overlap = concatenation_overlap_time)
                # parselmouth source code: static concatenate(sounds: List[parselmouth.Sound],
                # overlap: NonNegative[float] = 0.0)→ parselmouth.Sound
                # TODO: switch concatenation_overlap_time (in constants) if necessary
                story_sound.save(audio_story_path_out, 'WAV')
                all_audio_patientonly_files_list.append(audio_story_path_out)

            else:
                story_sound = parselmouth.Sound(audio_story_stroke_file_path)
                story_sound.save(audio_story_path_out, 'WAV')
                all_audio_patientonly_files_list.append(audio_story_path_out)
        # if the weekend story exists (and thus not the stroke story), set this one as the only story for that subject
        elif audio_story_weekend_file_path in all_audio_patientonly_nonmerged_files_list:
            story_sound = parselmouth.Sound(audio_story_weekend_file_path)
            story_sound.save(audio_story_path_out, 'WAV')

     # Now perform a double check whether all story-files have been merged (then called 'narrative')
    for audio_file in all_audio_patientonly_files_list:  # sanity check
        if 'story' in audio_file:  # now there shouldn't be any story file left, only 'narratives'
            print(
                f"Some story files haven't been merged yet: {audio_file}. "
                f"Audio_patientonly 'story' for this file has been generated but has not yet been merged with 'story_weekend' "
                f"nor has been checked whether this file is the only 'story' part for this subject (i.e., we don't know"
                f"whether story_weekend exists for this subject.")
            continue

    return all_audio_patientonly_files_list


if __name__ == "__main__":
    # raw_audio_path_in = os.path.join(AUDIO_DIR, 'sub-a043_story_stroke.wav')
    raw_dir = AUDIO_DIR
    diarization_dir = DIAR_DIR
    interim_dir= NONMERGED_AUDIO_PATIENTU_DIR
    processed_dir = AUDIO_PATIENTU_DIR
    # diar_txt_path_in = os.path.join(DIAR_DIR,'sub-a043_ANTAT.txt')

    # filter_audio_file_uninterruptedmerged(raw_audio_path_in, diarization_dir, processed_dir)
    # filter_audio_file(raw_audio_path_in, diarization_dir, processed_dir)
    # interim_dir = CLEAN_DIAR_DIR_DUMMY
    # cleanup_diar_txt_file(diar_txt_path_in, interim_dir)

    # preprocess_IANSA_audio_uninterruptedmerged(raw_dir, diarization_dir, interim_dir, processed_dir, overrule_spk_code_list=None)
    # preprocess_IANSA_audio(raw_dir, diarization_dir, interim_dir, processed_dir, overrule_spk_code_list=None)

    preprocess_IANSA_audio_uninterruptedmerged(raw_dir, diarization_dir, interim_dir, processed_dir,
                                               overrule_spk_code_list=(('sub-0222_CAT-NL', 0),

                                               ))
    """preprocess_IANSA_audio_uninterruptedmerged(raw_dir, diarization_dir, interim_dir, processed_dir,
                                               overrule_spk_code_list=(('sub-a006_CAT-NL', 0),
                                                                       ('sub-c060_MCA', 1),
                                                                       ('sub-c060_story_weekend', 0))
                                               )
    
    """




