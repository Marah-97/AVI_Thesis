# This function is to replace the 1's with 0's in the txt files.
# The txt files in #labels folder in each of #train #test #validation folders contain:
# one or more lines where the first element is the number of the defect type category, where ChipB would be 0.
# it will also delete the other classes than chip # the folder H cam5 has class 0 as chip. remember to change them before.
import os

def modify_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                with open(file_path, 'w') as file:
                    for line in lines:
                        parts = line.split()
                        if parts and parts[0] == '1':
                            parts[0] = '0'  # Change the first element to 0 if it's 1
                            new_line = ' '.join(parts) + '\n'
                            file.write(new_line)
                        # Lines where the first element is not 1 are not written back, effectively deleting them




# main_folder_path = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4\labels"
main_folder_path = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5\labels'
modify_files_in_folder(main_folder_path)
print("Folder --- is done changing class names")
