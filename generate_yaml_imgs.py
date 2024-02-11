import yaml
import os

# TODO:
# Exact small_imgs, mono, error, files via an algo
# filter imamges into diffrent types into diffrent YAML files.

root_folder_path = "dataset/coco-2017/"
destination_folder = "train/data/"

# Full path to the destination folder
full_path = os.path.join(root_folder_path, destination_folder)

# Dictionary to store file names
file_dict = {}

# List all files in the folder
for filename in os.listdir(full_path):
    file_path = os.path.join(full_path, filename)
    if os.path.isfile(file_path):
        file_dict[filename] = os.path.join(destination_folder, filename)
        print(f"File found: {filename}")

# Save to a YAML file
with open('file_list.yaml', 'w') as file:
    yaml.dump(file_dict, file)

print("File list saved to file_list.yaml")