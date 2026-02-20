""" Script to find text files matching to audio files """


import os
import shutil

# Define folder paths
audio_folder = r"C:\Users\u0146803\git\Ella_PhD_NLP_repo\data\raw\raw_audio_febr2026"
text_folder = r"C:\Users\u0146803\git\Ella_PhD_NLP_repo\data\raw\raw_docx_febr2026"
output_folder = r"C:\Users\u0146803\git\Ella_PhD_NLP_repo\data\interim\txt_transcripts_preclean_febr2026"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all audio file base names (without extension)
audio_files = [
    os.path.splitext(f)[0]
    for f in os.listdir(audio_folder)
    if f.lower().endswith(".wav")
]

print(f"Found {len(audio_files)} audio files.")

copied_count = 0
skipped_count = 0
missing_count = 0

# Loop through audio files and match text files
for base_name in audio_files:
    txt_filename = base_name + ".txt"
    source_txt_path = os.path.join(text_folder, txt_filename)
    destination_txt_path = os.path.join(output_folder, txt_filename)

    if os.path.exists(source_txt_path):
        # Check if file already exists in destination
        if os.path.exists(destination_txt_path):
            # print(f"Skipped (already exists): {txt_filename}")
            skipped_count += 1
        else:
            shutil.copy2(source_txt_path, destination_txt_path)
            # print(f"Copied: {txt_filename}")
            copied_count += 1
    else:
        print(f"Missing text file for: {base_name}")
        missing_count += 1

print("\n--- Summary ---")
print(f"Copied:  {copied_count}")
print(f"Skipped: {skipped_count} (already existed)")
print(f"Missing: {missing_count}")
print("Done.")