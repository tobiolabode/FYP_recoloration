import yaml
import os

# TODO:
# Exact small_imgs, mono, error, files via an algo
# filter imamges into diffrent types into diffrent YAML files.

root_folder_path = "C:\\Users\\tnint\\Coding\\Side_projects\\GitHub_Export\\spcolor\\dataset\\"
destination_folder = "imgs\\train\\data\\"
# C:\Users\tnint\Coding\Side_projects\GitHub_Export\spcolor\dataset\coco-2017\imgs\train\data
# Full path to the destination folder
# full_path = os.path.join(root_folder_path, destination_folder)

hard_coded_path = "C:\\Users\\tnint\\Coding\\Side_projects\\GitHub_Export\\spcolor\\dataset\\coco-2017\\imgs\\train\\data\\"

full_path = hard_coded_path

# Dictionary to store file names
file_dict = {}

# List all files in the folder
for filename in os.listdir(full_path):
    file_path = os.path.join(full_path, filename)
    if os.path.isfile(file_path):
        file_dict[filename] = os.path.join(destination_folder, filename)
        print(f"File found: {filename}")

# Save to a YAML file
with open('file_list_2_img.yaml', 'w') as file:
    yaml.dump(file_dict, file)

print("File list saved to file_list.yaml")