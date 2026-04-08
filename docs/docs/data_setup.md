# Data Setup

All original data must be placed in the `Data/RAW` directory.

Folder names must include a suffix `_monthYear`
(e.g. `raw_audio_Mar2026`).

---

## Required RAW Folders

Inside `Data/RAW` create:

- `diarization_monthYear`
  - Speaker diarization outputs

- `docx_transcripts_monthYear`
  - Original transcripts in `.docx` format

- `raw_audio_monthYear`
  - Original `.wav` audio files

---

## Starting From TXT Files

If transcripts already exist as `.txt` files:

1. Place them in: Data/INTERIM/txt_transcripts_preclean_monthYear
2. 2. Use:
```bash
preprocess_transcripts_startfromtxtfile.py
```

## INTERIM and PROCESSED
All folder names used in RAW must be mirrored in:

- Data/INTERIM
- Data/PROCESSED

Folders must exist before running scripts.

Example INTERIM folders
Data/INTERIM/
├── audio_patientonlyU_nonmerged_monthYear
├── diarization_clean_monthYear   (only if needed)
└── txt_transcripts_preclean_monthYear

Example PROCESSED folders
Data/PROCESSED/
├── audio_patientonlyU_monthYear
└── txt_transcripts_monthYear


## Configure Constants
Before running any scripts, open:
ella_phd_nlp_code/data/CONSTANTS.py

Make sure:

Folder names match your _monthYear naming
Paths correctly point to Data/RAW, Data/INTERIM, and Data/PROCESSED

## Next step
**preprocessing**
