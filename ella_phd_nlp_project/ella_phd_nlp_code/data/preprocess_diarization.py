""" Preprocess diarization files
---------------------------------
This includes
- assigning speaker roles in diarization output files, based on the following rules:
        1. Original assumption: the FIRST speaker in the file is the test administrator.
        2. Correction rule: if a speaker has MORE TURNS (more diarization lines),
           that speaker is chosen as the patient.
        3. Tie-break rule: if both speakers have the same number of turns,
           fall back to the original assignment.
- checking assigned speaker roles and flagging suspicious diarization files for manual check;
"""

## Import statements
# ----------------------------------

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
    concatenation_overlap_time, TABLES_DIR)

from ella_phd_nlp_project.ella_phd_nlp_code.features.preliminary_analysis import *
from ella_phd_nlp_project.ella_phd_nlp_code.features.feature_helper_functions.helper_extraction_audio import *



## Code
# ------------------------------------

"""NOT USED: Create a helper function that checks diarization_files
# IF NEEDED: add interim_folder to store cleaned diarization files
def cleanup_diar_txt_file(diar_txt_path_in, interim_dir):
    
    Helper function to clean up the diar txt file
    :param diar_txt_path_in: the diarization.txt path (showing the indices per speaker: start, end, spk) that must
    be cleaned up
    :param interim_dir: the interim directory path where the diar_txt file must be temporarily saved
    :return: the text file in the interim directory, subject to the following changes
            1) only keep spk 0 and 1
    
    ## Choose the name for the cleaned diar_text file
    txt_name_subparts = os.path.splitext(os.path.basename(diar_txt_path_in))[0].split('_')
    # 'os.path.splitext(os.path.basename(diar_txt_path_in))[0]' extracts the name of the diar_txt_path
    # (all without the .txt extension)
    diar_txt_correct_name = '_'.join(txt_name_subparts)
    # make sure you only keep the 'sub-XXX', 'transcriptie' and 'NAME TASK' (so discard the last 'preclean' part) and
    # join them
    diar_txt_path_out = os.path.join(
        interim_dir, ".".join(
            ["_".join([str(diar_txt_correct_name), 'diarization']),
             'txt']))

    with (open(diar_txt_path_in, 'r') as infile, open(diar_txt_path_out, 'w') as outfile):
        # 'r' = open the txt_path_in file for reading, 'w' = open the txt_path_out for writing
        # Note: encoding as utf-8 is needed as Python= , encoding = 'utf-8'
        lines = infile.readlines()
        if not lines:
            raise ValueError("The input file is empty")

        for line in lines:
            # Remove lines ending with nbr 2, because we cannot have more than 2 speakers
            if line.endswith('2'):
                continue

            # Write the modified line to the output file, IF the line is not empty
            if line:
                outfile.write(line)

    return diar_txt_path_out
"""


## Create a helper function that returns the binary speaker code of the patient
def give_patient_spk_code(diar_txt_path_in):
    """
    Helper function to return the binary speaker code of the patient
    :param diar_txt_path_in: the diarization.txt path (showing the indices per speaker: start, end, spk)
    :return: the binary speaker code of the patient (0 or 1)

    Determine the binary speaker code for the patient based on:
        1. Original assumption: the FIRST speaker in the file is the test administrator.
        2. Correction rule: if a speaker has MORE TURNS (more diarization lines),
           that speaker is chosen as the patient.
        3. Tie-break rule: if both speakers have the same number of turns,
           fall back to the original assignment.

    If the assumptions don't count, then this function will not work and spk_code must be inputted manually:
    - to filter ONE audio file: add as parameters in 'filter_audio_file' function: new_spk_code (number 0 or 1 or another), overrule_spk_code = True
    - to filter ALL audio files with some in need of manually added speaker code: add as parameter in
    'preprocess_IANSA_audio': overrule_spk_code_list = list(tuplex, tupley), with each tuple = (audio-file name (e.g.,
    sub-a043_ANTAT), correct speaker code)
    Note: if diarization_files must be cleaned, then param diar_txt_path_in should be diar_txt_path_in_cleaned (as you
    will use the cleaned diarization files)
    """

    # ------------ PART 1: READ DIARIZATION FILE ------------
    segments = []
    with open(diar_txt_path_in, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            start, end, spk = line.split(',')
            segments.append((float(start), float(end), int(spk)))

    if not segments:
        raise ValueError("Diarization file is empty.")

    # ------------ PART 2: ORIGINAL ASSIGNMENT ------------
    first_spk = segments[0][2]
    distinct_spks = {seg[2] for seg in segments}

    if len(distinct_spks) == 1:
        # Only one speaker → must be patient
        original_patient_code = first_spk
    else:
        # First speaker = test administrator → patient is the other one
        original_patient_code = 1 - first_spk

    # ------------ PART 3: COUNT TURNS PER SPEAKER ------------
    turn_count = {0: 0, 1: 0}

    for _, _, spk in segments:
        turn_count[spk] += 1

    # ------------ PART 4: DECISION LOGIC WITH TIE-BREAK ------------
    if turn_count[0] > turn_count[1]:
        final_patient_code = 0
    elif turn_count[1] > turn_count[0]:
        final_patient_code = 1
    else:
        # Tie → use original assumption
        final_patient_code = original_patient_code

    return final_patient_code


""" OLD FUNCTION (did not count total turns)
def give_patient_spk_code(diar_txt_path_in):
    first_line_diar_code = int()
    spk_code = int()
    with open(diar_txt_path_in, 'r') as f:
        first_line = f.readline().strip()  # Read only the first line and strip whitespace

        # line format is: "start,end,speaker" -> we split by ','
        parts = first_line.split(',')

        # Get the last element, which should be the speaker code
        first_line_diar_code = int(parts[-1])

        lines = f.readlines()
        if not lines:  # there was only one line
            spk_code = first_line_diar_code
        else:
            for line in lines:
                parts = line.split(',')
                diar_code_this_line = int(parts[-1])
                if diar_code_this_line != first_line_diar_code:  # so there are multiple speakers
                    if first_line_diar_code == 0:  # we assume first line will be the test administrator, so the first spk is NOT who we want
                        spk_code = int(1)  # if the first speaker (test admin) has spk-code 0, then the patient has code 1...
                    if first_line_diar_code == 1:
                        spk_code = int(0)  #... and vice versa
                    break
                else:  # if only one speaker appears, then it will be the patient
                    spk_code = first_line_diar_code

    return int(spk_code)
    """


## Check the diarization assignment output
def sanity_check_diarization(
        diar_txt_path,
        file_id=None,
        long_initial_threshold=4.0,
        avg_duration_min_threshold=0.05,
        consecutive_first_spk_threshold=5):
    """
    Performs a sanity check on diarization files.

    Rules included:
    - long initial segment rule
    - over-segmentation rule (segments too short)
    - only one speaker label rule
    - second speaker (expected patient) does NOT have the most turns rule
    - first speaker (expected test administrator) has many initial consecutive turns rule

    Returns:
    - dictionary with diagnostic info and list of issues
    """

    issues = []
    segments = []

    # ---------------- READ DIARIZATION FILE ----------------
    with open(diar_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            start, end, spk = line.split(",")
            segments.append((float(start), float(end), int(spk)))

    if not segments:
        return {
            "file": file_id,
            "issues": ["empty_file"],
            "recommended_manual_check": True
        }

    # ---------------- BASIC STATS ----------------
    durations = [end - start for start, end, _ in segments]
    avg_duration = sum(durations) / len(durations)
    max_duration = max(durations)

    segment_stats = {
        "avg_duration": avg_duration,
        "max_duration": max_duration,
        "total_segments": len(segments)
    }

    # ---------------- ONLY ONE SPEAKER LABEL ----------------
    speaker_labels = {spk for _, _, spk in segments}  # note: a set removes duplicates so if consecutive speaker 0, then set = {0}
    if len(speaker_labels) == 1:
        issues.append("only_one_speaker_detected")

    # ---------------- LONG INITIAL SEGMENT ----------------
    first_start, first_end, first_spk = segments[0]
    if (first_end - first_start) > long_initial_threshold:
        issues.append("long_initial_segment")

    # ---------------- OVER-SEGMENTATION ----------------
    if avg_duration < avg_duration_min_threshold:
        issues.append("oversegmentation_possible")

    # ---------------- TURN COUNTS ----------------
    turn_counts = {0: 0, 1: 0}
    for _, _, spk in segments:
        if spk in (0, 1):
            turn_counts[spk] += 1
    # second speaker (patient expected based on your logic)
    # = opposite of first_spk
    expected_patient = 1 - first_spk

    # ---------------- SECOND SPEAKER NOT MOST TURNS ----------------
    if turn_counts[expected_patient] < turn_counts[first_spk]:
        issues.append("expected_patient_not_most_turns")

    # ---------------- FIRST SPEAKER MANY INITIAL CONSECUTIVE TURNS ----------------
    consecutive = 0
    for _, _, spk in segments:
        if spk == first_spk:
            consecutive += 1
        else:
            break

    if consecutive >= consecutive_first_spk_threshold:
        issues.append("first_speaker_many_initial_consecutive_turns")

    # ---------------- OUTPUT ----------------
    return {
        "file": file_id,
        "issues": issues,
        "turn_counts": turn_counts,
        "segment_stats": segment_stats,
        "recommended_manual_check": len(issues) > 0
    }


## MAIN LOOP: process folder & save Excel
# ---------------------------------------------------------

def process_all_diarizations(
        diarization_dir,
        tables_dir,
        output_filename="diarization_report.xlsx"):
    """

    :param diarization_dir: the directory containing the diarization files (= output of ecapa diarization script)
    :param tables_dir: the path where the diarization report will be saved
    :param output_filename: the name of the output file
    :return:  a single Excel file with:
    | file_id | patient_speaker_code | sanity_check_issues |
    (issues stored as a comma‑separated string or list)

    How it works:
    - Loops through all diarization .txt files in the diarization folder
    - Computes the assigned patient speaker code (using the logic from the give_patient_spk_code function)
    - Runs the sanity check (using the sanity_check_diarization function)
    - Saves the results in a single Excel file with:| file_id | patient_speaker_code | sanity_check_issues |
    (issues stored as a comma‑separated string or list)
    """

    # Ensure output directory exists
    os.makedirs(tables_dir, exist_ok=True)

    # Build full output path
    output_excel_path = os.path.join(tables_dir, output_filename)

    rows = []

    for filename in os.listdir(diarization_dir):
        if filename.endswith(".txt"):
            diar_path = os.path.join(diarization_dir, filename)
            file_id = os.path.splitext(filename)[0]

            # Get speaker code
            spk_code = give_patient_spk_code(diar_path)

            # Sanity check
            sanity = sanity_check_diarization(diar_path, file_id=file_id)
            issues_text = ", ".join(sanity["issues"]) if sanity["issues"] else ""

            rows.append({
                "file_id": file_id,
                "patient_speaker_code": spk_code,
                "sanity_check_issues": issues_text
            })

    # Save to Excel
    df = pd.DataFrame(rows)
    df.to_excel(output_excel_path, index=False)

    print(f" Diarization report saved to: {output_excel_path}")
    return df



if __name__ == "__main__":
    diarization_dir = DIAR_DIR
    # diar_txt_path_in = os.path.join(DIAR_DIR,'sub-a043_ANTAT.txt')
    tables_dir = TABLES_DIR

    process_all_diarizations(
        diarization_dir,tables_dir)

