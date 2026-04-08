Getting started
===============

# Getting Started (Command Overview)

Run the following commands **from the repository root** in the **Anaconda Prompt**.

For full explanations, see each markdown file:
-> installation
-> data_setup
-> preprocessing
-> dataframe_creation
---

## 1. Set up and activate the environment

```cmd
conda env create -f environment.yml
conda activate Ella_PhD_NLP
```

## 2. set up data
Place all original, untouched files in Data/RAW


Create subfolders using a _monthYear naming convention, for example:

raw_audio_Mar2026
diarization_Mar2026
docx_transcripts_Mar2026



Create matching (empty) folders in:

Data/INTERIM
Data/PROCESSED

Example INTERIM folders
Data/INTERIM/
├── audio_patientonlyU_nonmerged_monthYear
├── diarization_clean_monthYear   (only if needed)
└── txt_transcripts_preclean_monthYear

Example PROCESSED folders
Data/PROCESSED/
├── audio_patientonlyU_monthYear
└── txt_transcripts_monthYear



Update folder names and paths in CONSTANTS.py so they match your setup exactly


## 3. Run preprocessing steps (in this exact order)
2.1 Preprocess diarization
```cmd
python ella_phd_nlp_code\data\preprocess_diarization.py
```
(Review the diarization report before continuing.)

2.2 Preprocess audio
```cmd
python ella_phd_nlp_code\data\preprocess_audio.py
```

2.3 Preprocess transcripts
Run one of the following:
If starting from DOCX files:
```cmd
python ella_phd_nlp_code\data\preprocess_transcripts.py
```
If starting from TXT files:
```cmd 
python ella_phd_nlp_code\data\preprocess_transcripts_startfromtxtfile.py
```

## 3. Create feature tables (dataframes)
```cmd
mkdir reports\tables\tables_monthYear
python ella_phd_nlp_code\features\feature_implementation\make_datafram
```

Output
Feature tables will be saved in:
reports/tables/tables_monthYear/