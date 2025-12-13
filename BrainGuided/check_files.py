import os
import csv

# Set the path to the directory you want to check
directory_path = 'EK6--raw'

# Set the path to the CSV file
csv_file_path = 'video_id_ek6_original.csv'

# Read the CSV file and get the list of file names
csv_file_names = []
with open(csv_file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)  # Skip the header row
    for row in reader:
        csv_file_names.append(row[1])  # Assuming the file names are in the first column

directory_files = []
# Get the list of files in the directory
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        if file_name.endswith(".mp4") or file_name.endswith(".avi"):
            video_name = os.path.splitext(file_name)[0]
            directory_files.append(os.path.join(os.path.split(root)[1], video_name))

# Find the files in the directory that are not in the CSV file
missing_files = [file for file in directory_files if file not in csv_file_names]
missing_files_csv = [file for file in csv_file_names if file not in directory_files]

# Print the missing files
if missing_files:
    print("The following files are in the directory but not in the CSV file:")
    for file in missing_files:
        print(file)
else:
    print("All files in the directory are present in the CSV file.")

# Print the missing files of csv
if missing_files_csv:
    print("The following files are in the CSV file but not in the directory:")
    for file in missing_files_csv:
        print(file)
else:
    print("All files in the CSV file are present in the directory.")