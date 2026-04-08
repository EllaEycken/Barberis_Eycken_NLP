# Preprocessing Pipeline

This page describes the full preprocessing pipeline used in the project.
All scripts must be run **from the repository root** and **in the documented order**.

Before starting, make sure that:

- The Conda environment is activated
- All required data folders exist
- Folder names in `CONSTANTS.py` are correct

---

## Script Execution Order (Summary)
The preprocessing steps must be run in this order:

- preprocess_diarization.py
- preprocess_audio.py
- preprocess_transcripts.py OR preprocess_transcripts_startfromtxtfile.py

Running scripts out of order may result in missing or incorrect outputs.

## Before You Start

Activate the Conda environment:

```bash
conda activate Ella_PhD_NLP
```
Verify that Python points to the correct environment:
```bash
which python
```
Expected output contains:
.../anaconda3/envs/Ella_PhD_NLP/python

## Step 1: Preprocess Diarization
```bash
python ella_phd_nlp_code/data/preprocess_diarization.py``Show more lines
```
### Purpose
This script assigns speaker roles (patient vs test administrator) based on
the diarization output files.

### Speaker Assignment Logic
The following rules are applied in order:
1. Default assumption: The first speaker in the diarization file is assigned as the test administrator.
2. Correction rule: If one speaker has more turns (more diarization segments),
that speaker is assigned as the patient.
3. Tie‑break rule: If both speakers have the same number of turns,
the default assignment is retained.

### Output
Cleaned diarization files
A diarization report identifying suspicious cases
!!! note
Always inspect the diarization report before continuing.
Some files may require manual correction of speaker roles.


## Step 2: Preprocess Audio
```bash
python ella_phd_nlp_code/data/preprocess_audio.pyShow more lines
```
### Purpose
This script uses the diarization output to generate patient‑only audio files.
What the script does

- Applies diarization timestamps to the original audio
- Removes speech not produced by the patient
- Creates diarized, patient‑only audio files

!!! warning
Make sure diarization corrections from the previous step have been applied
before running this script. Incorrect speaker roles will lead to incorrect
audio segmentation.

## Step 3: Preprocess Transcripts
Transcript preprocessing depends on the input format.
### Option A: Starting from DOCX files
```bash
python 
ella_phd_nlp_code/data/preprocess_transcripts.pyShow more lines
```

### Option B: Starting from TXT files
```bash
python 
ella_phd_nlp_code/data/preprocess_transcripts_startfromtxtfile.pyShow more lines
```

### Purpose
These scripts:

- Clean formatting artifacts
- Standardize transcript structure
- Prepare text files for downstream feature extraction

!!! note
Only run one of the transcript preprocessing scripts,
depending on your input format.

## Next step:
**dataframe_creation.md**