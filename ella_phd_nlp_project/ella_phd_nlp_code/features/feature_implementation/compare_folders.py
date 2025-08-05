""" Compare items in folders"""


import os

def all_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

folder1 = r'C:\Users\u0146803\git\Ella_PhD_NLP_repo\data\processed\audio_patientonlyU'
folder2 = r'C:\Users\u0146803\git\Ella_PhD_NLP_repo\data\processed\txt_transcripts'

files1 = all_paths(folder1)
files2 = all_paths(folder2)

def extract_subject_question_names(file_list):
    result = []
    for file in file_list:
        file_name = os.path.basename(file)
        splitted_file_name = file_name.split("_")
        subject_name = splitted_file_name[0]
        question_name = splitted_file_name[2].split(".")[0]
        result.append((subject_name, question_name))  # Using tuple instead of list
    return result

list1 = extract_subject_question_names(files1)
list2 = extract_subject_question_names(files2)

set1 = set(list1)
set2 = set(list2)

only_in_folder1 = set1 - set2
only_in_folder2 = set2 - set1

print("Files only in Folder 1:")
print("\n".join(f"{s}_{q}" for s, q in sorted(only_in_folder1)))

print("\nFiles only in Folder 2:")
print("\n".join(f"{s}_{q}" for s, q in sorted(only_in_folder2)))
