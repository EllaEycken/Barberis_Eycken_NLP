# Dataframe Creation

This page describes how to generate **feature tables (dataframes)** from the
preprocessed audio and transcript data.

Make sure that **all preprocessing steps have completed successfully**
before running the dataframe creation script.

---

## Before You Start

Ensure that:

- The Conda environment is activated
- Preprocessed data exists in `Data/PROCESSED`
- Folder names in the code match your `_monthYear` naming

Activate the environment if needed:

```bash
conda activate Ella_PhD_NLP
```

## Step 1: Create the Output Folder
Before generating dataframes, create a folder where the output tables will be stored.
From the repository root:
```bash
mkdir reports/tables/tables_monthYear
```
Replace tables_monthYear with your chosen name
(e.g. tables_Mar2026).

## Step 2: Generate Feature Dataframes
Run the dataframe creation script from the repository root:
```bash
python ella_phd_nlp_code/features/feature_implementation/make_dataframe.pyShow more lines
```

## What This Script Does
The dataframe creation script:

- Loads preprocessed audio and transcript data
- Extracts linguistic and acoustic features per patient
- Aggregates features into structured dataframes
- Saves the resulting tables to the reports directory

The exact set of features depends on the implementation inside
make_dataframe.py.

## Output Location
The generated tables are saved in:
reports/tables/tables_monthYear/

Each file typically corresponds to:

- A specific feature set, or
- A specific aggregation level (e.g. per task)


## Verifying the Output
After running the script, check that:

- The output folder exists
- The folder is not empty
- File names and formats are as expected
- Tables contain reasonable values

If tables are empty or missing expected columns, verify that:

- Preprocessing completed successfully
- Input paths inside the code are correct