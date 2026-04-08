## 🩺 `docs/troubleshooting.md`


# Troubleshooting

This page lists common issues encountered when running the pipeline, along with
suggested solutions.

If an error occurs, always inspect the full error message carefully—most issues
are caused by incorrect paths or missing folders.

---

## Environment Issues

### Conda environment not activated

**Symptoms**
- `ModuleNotFoundError`
- Unexpected Python or package versions

**Solution**

Activate the correct Conda environment:

```bash
conda activate Ella_PhD_NLP
```

Verify that the correct Python interpreter is being used:
```bash
which python
```
The path should include:
anaconda3/envs/Ella_PhD_NLP


## Path and Folder Errors
### FileNotFoundError
Common causes

Folder names do not match the _monthYear naming convention
Required INTERIM or PROCESSED folders do not exist
Paths in CONSTANTS.py do not correspond to actual folders

Checklist

Folder exists at the specified location
Folder name matches exactly (including capitalization)
Folder paths in the code are correct


### Scripts run without producing output
Possible causes

Scripts run in the wrong order
Input folders are empty
Output folders were not created in advance

Solution

Follow the documented pipeline order strictly
Verify input data is present
Create all required output folders before running scripts


## Diarization‑Related Issues
### Incorrect speaker assignment
Symptoms

Patient speech missing
Administrator speech included in patient output

Cause

Automatic speaker assignment failed for a given file

Solution

Inspect the diarization report
Manually correct flagged speaker assignments
Re‑run preprocess_audio.py after corrections


## Transcript‑Related Issues
### Transcript preprocessing fails
Common causes

Mixed input formats (DOCX and TXT)
Incorrect preprocessing script used

Solution

Use preprocess_transcripts.py for DOCX files
Use preprocess_transcripts_startfromtxtfile.py for TXT files
Only run one transcript preprocessing script per dataset


## Dataframe Issues
### Empty or incomplete dataframes
Possible causes

Missing data in Data/PROCESSED
Feature extraction failed silently
Incorrect paths in make_dataframe.py

Steps to debug

Confirm that PROCESSED data exists
Re‑run preprocessing steps if needed
Add temporary print or logging statements to inspect intermediate outputs


## General Debugging Advice

Run scripts one at a time
Inspect outputs after each step
Never modify files in Data/RAW
Re‑export environment.yml if new dependencies are added


## When Problems Persist
If an issue cannot be resolved:

Copy the full error message
Note which script was running
Check recent changes to data or paths
Contact the repository maintainer or open an issue

Clear and complete error descriptions make troubleshooting much easier.