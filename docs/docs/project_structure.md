# Project Structure

The repository is organized as follows:
Ella_PhD_NLP_repo/
│
├── ella_phd_nlp_code/
│   ├── data/
│   │   ├── preprocess_diarization.py
│   │   ├── preprocess_audio.py
│   │   ├── preprocess_transcripts.py
│   │   └── preprocess_transcripts_startfromtxtfile.py
│   │
│   └── features/
│       └── feature_implementation/
│           └── make_dataframe.py
│
├── Data/
│   ├── RAW/
│   ├── INTERIM/
│   └── PROCESSED/
│
├── reports/
│   └── tables/
│
├── environment.yml
├── requirements.txt
└── docs/

## Key Distinction

- `ella_phd_nlp_code/` → **Code**
- `Data/` → **All datasets**
- `reports/` → **Analysis outputs**

Never place scripts inside the `Data` directory.