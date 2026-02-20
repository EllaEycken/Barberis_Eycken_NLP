Generating the docs
----------

Use [mkdocs](http://www.mkdocs.org/) structure to update the documentation. 

Build locally with:

    mkdocs build

Serve locally with:

    mkdocs serve


**How to use this NLP_repo:**
1) Load original data in data folder RAW by making the following subfolders (add _monthYear depending on when you're performing your analyses) and store the appropriate files there:
   1) In 'diarization_monthYear': store diarization files of the original audio files, based on the diarization script (diarize.py, see L-drive)
   2) In 'docx_transcripts_monthYear': store the original docx files (if it is docx files)
      1) Note: if you start with text files, store them in the INTERIM_DIR (under the folder 'txt_transcripts_preclean_monthYear' and then run preprocess_transcripts_startfromtxtfile.py)
   3) In 'raw_audio_monthYear': store audio files (.wav)
2) First, PREPROCESS the files
   1) First run preprocess_audio.py. This will make sure the original audio files are transformed in diarized audio files (filtering for the patient's voice)
   2) Then run preprocess_transcripts.py . This will ensure the original transcripts (word files) are nicely structured for further analyses.
      1) Note: if you start from text-files instead of word files, run preprocess_transcripts_startfromtxtfile.py instead.
3) Then, make the DATAFRAMES to visualize tables of feature values per patient. 
   1) Do this by first making a folder 'tables_monthYear' in Reports-->tables (where the table will be stored) 
   2) Then adapt the paths and run make_dataframe.py
4) The dataframes will be found in reports-->tables--> tables_monthYear
