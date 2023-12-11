# create these empty folders, and move the images into #images folder, labels into #labels folder, then the path is the DataFolder, run the function.
# - DataFolder
#     - images
#         - train    - test    - validation
#     - labels
#         - train    - test    - validation
###################################################################################################################
###################################################################################################################
## This function consider the split wrt. vialID and splitting also the labels (.txt files)

import os
import shutil
import collections

def divide_data_by_vials(data_folder, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Create subdirectories if they don't exist
    subfolders = ['train', 'validation', 'test']
    for folder in ['images', 'labels']:
        for subfolder in subfolders:
            ensure_dir(os.path.join(data_folder, folder, subfolder))

    # Group files by vial ID
    def list_files_and_group(folder, file_exts):
        files = [f for f in os.listdir(os.path.join(data_folder, folder)) 
                 if os.path.isfile(os.path.join(data_folder, folder, f)) 
                 and f.lower().endswith(file_exts)]
        vials = collections.defaultdict(list)
        for file in files:
            vial_id = file.split('vialID')[1].split('-')[0]  # vial_id = file.split('-')[1] 
            vials[vial_id].append(file)
        return vials

    image_groups = list_files_and_group('images', ('.png', '.jpg', '.jpeg'))
    label_groups = list_files_and_group('labels', ('.txt',))

    # Ensure the same set of vial IDs in both images and labels
    common_vials = set(image_groups.keys()).intersection(set(label_groups.keys()))
    image_groups = {k: image_groups[k] for k in common_vials}
    label_groups = {k: label_groups[k] for k in common_vials}

    # Divide and assign vials to train, validation, and test sets
    total_vials = len(common_vials)
    num_train_vials = int(total_vials * train_ratio)
    num_valid_vials = int(total_vials * valid_ratio)

    # Function to move files
    def move_files(file_groups, dest_folder, data_type):
        for _, file_list in file_groups.items():
            for file_name in file_list:
                src_path = os.path.join(data_folder, data_type, file_name)
                dest_path = os.path.join(data_folder, data_type, dest_folder, file_name)
                shutil.move(src_path, dest_path)

    # Splitting vial IDs into train, validation, and test sets
    vial_ids = list(common_vials)
    train_vials = set(vial_ids[:num_train_vials])
    valid_vials = set(vial_ids[num_train_vials:num_train_vials + num_valid_vials])
    test_vials = set(vial_ids[num_train_vials + num_valid_vials:])

    # Moving files to respective folders
    for vial_id in train_vials:
        if vial_id in image_groups:
            move_files({vial_id: image_groups[vial_id]}, 'train', 'images')
            move_files({vial_id: label_groups[vial_id]}, 'train', 'labels')
    for vial_id in valid_vials:
        if vial_id in image_groups:
            move_files({vial_id: image_groups[vial_id]}, 'validation', 'images')
            move_files({vial_id: label_groups[vial_id]}, 'validation', 'labels')
    for vial_id in test_vials:
        if vial_id in image_groups:
            move_files({vial_id: image_groups[vial_id]}, 'test', 'images')
            move_files({vial_id: label_groups[vial_id]}, 'test', 'labels')

    print("Data has been divided into train, validation, and test folders by vial groups.")


divide_data_by_vials(r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5")

###################################################################################################################
###################################################################################################################
## This function to do the split only for non-defectives ones (images without labels)

import os
import shutil

def divide_images_by_vials(data_folder, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Create subdirectories for images if they don't exist
    subfolders = ['train', 'validation', 'test']
    for subfolder in subfolders:
        ensure_dir(os.path.join(data_folder, 'images', subfolder))

    # Group images by vial ID
    def list_images_and_group():
        image_folder = os.path.join(data_folder, 'images')
        files = [f for f in os.listdir(image_folder) 
                 if os.path.isfile(os.path.join(image_folder, f)) 
                 and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        vials = collections.defaultdict(list)
        for file in files:
            vial_id = file.split('vialID')[1].split('-')[0]  
            vials[vial_id].append(file)
        return vials

    image_groups = list_images_and_group()

    # Divide and assign vials to train, validation, and test sets
    total_vials = len(image_groups)
    num_train_vials = int(total_vials * train_ratio)
    num_valid_vials = int(total_vials * valid_ratio)

    # Function to move images
    def move_images(file_groups, dest_folder):
        for _, file_list in file_groups.items():
            for file_name in file_list:
                src_path = os.path.join(data_folder, 'images', file_name)
                dest_path = os.path.join(data_folder, 'images', dest_folder, file_name)
                shutil.move(src_path, dest_path)

    # Splitting vial IDs into train, validation, and test sets
    vial_ids = list(image_groups.keys())
    train_vials = set(vial_ids[:num_train_vials])
    valid_vials = set(vial_ids[num_train_vials:num_train_vials + num_valid_vials])
    test_vials = set(vial_ids[num_train_vials + num_valid_vials:])

    # Moving images to respective folders
    for vial_id in train_vials:
        move_images({vial_id: image_groups[vial_id]}, 'train')
    for vial_id in valid_vials:
        move_images({vial_id: image_groups[vial_id]}, 'validation')
    for vial_id in test_vials:
        move_images({vial_id: image_groups[vial_id]}, 'test')

    print("Images have been divided into train, validation, and test folders by vial groups.")

divide_images_by_vials(r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3")


