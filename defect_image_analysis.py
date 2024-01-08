# This file is to generate the plots to the data analysis section

import os
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

data_folder1 = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3_1050'
data_folder2 = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050'
data_folder3 = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5_1050'
#################################################################################
# function to count the defect images per vial

def analyze_defects(data_folder):
    defect_counts = defaultdict(int)
    vials_with_defects = set()
    subfolders = ['train', 'test', 'validation']

    for subfolder in subfolders:
        image_subfolder = os.path.join(data_folder, 'images', subfolder)
        label_subfolder = os.path.join(data_folder, 'labels', subfolder)

        for image_file in os.listdir(image_subfolder):
            vial_id = image_file.split('vialID')[1].split('-')[0]
            label_file_path = os.path.join(label_subfolder, image_file.replace('.jpg', '.txt'))

            if os.path.exists(label_file_path) and os.path.getsize(label_file_path) > 0:
                defect_counts[vial_id] += 1
                vials_with_defects.add(vial_id)

    # Filter out vials that do not have any defects
    defect_counts = {vial: count for vial, count in defect_counts.items() if vial in vials_with_defects}

    return defect_counts

#################################################################################
## Histogram of Defect Images per Vial

def plot_histogram(defect_counts):
    counts = list(defect_counts.values())

    plt.figure(figsize=(10, 5))
    plt.hist(counts, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Number of Defect Images per Vial')
    plt.ylabel('Frequency')
    plt.title('Histogram of Defect Images per Vial')
    plt.xlim(0, 30)  # Setting x-axis range from 0 to 30
    plt.show()

# Analyze defects
defect_counts1 = analyze_defects(data_folder1)
defect_counts2 = analyze_defects(data_folder2)
defect_counts3 = analyze_defects(data_folder3)

# Plot histogram
plot_histogram(defect_counts1)
plot_histogram(defect_counts2)
plot_histogram(defect_counts3)

#################################################################################
## Find and print the minimum number of defect images per vial

if defect_counts1:
    min_defects = min(defect_counts1.values())
    max_defects = max(defect_counts1.values())
    print(f"Minimum number of defect images per vial in CAM 3: {min_defects}")
    print(f"Maximum number of defect images per vial in CAM 3: {max_defects}")    
else:
    print("No defect images found.")

if defect_counts2:
    min_defects = min(defect_counts2.values())
    max_defects = max(defect_counts2.values())
    print(f"Minimum number of defect images per vial in CAM 4: {min_defects}")
    print(f"Maximum number of defect images per vial in CAM 4: {max_defects}")    
else:
    print("No defect images found.")
    
if defect_counts3:
    min_defects = min(defect_counts3.values())
    max_defects = max(defect_counts3.values())
    print(f"Minimum number of defect images per vial in CAM 5: {min_defects}")
    print(f"Maximum number of defect images per vial in CAM 5: {max_defects}")    
else:
    print("No defect images found.")
#################################################################################
## Boxplot of Defect Images per Vial for The three Camera Data

def analyze_defects2(data_folder):
    defect_counts = defaultdict(int)
    subfolders = ['train', 'test', 'validation']

    for subfolder in subfolders:
        image_subfolder = os.path.join(data_folder, 'images', subfolder)
        label_subfolder = os.path.join(data_folder, 'labels', subfolder)

        for image_file in os.listdir(image_subfolder):
            vial_id = image_file.split('vialID')[1].split('-')[0]
            label_file_path = os.path.join(label_subfolder, image_file.replace('.jpg', '.txt'))

            if os.path.exists(label_file_path) and os.path.getsize(label_file_path) > 0:
                defect_counts[vial_id] += 1

    return list(defect_counts.values())

def plot_boxplots(defect_counts1, defect_counts2, defect_counts3):
    plt.figure(figsize=(10, 5))
    plt.boxplot([defect_counts1, defect_counts2, defect_counts3])
    plt.ylabel('Number of Defect Images per Vial')
    plt.title('Boxplot of Defect Images per Vial for The three Camera Data')
    plt.xticks([1, 2, 3], ['CAM 3 Data', 'CAM 4 Data', 'CAM 5 Data'])
    plt.show()


# Analyze defects for each folder
defect_counts1 = analyze_defects2(data_folder1)
defect_counts2 = analyze_defects2(data_folder2)
defect_counts3 = analyze_defects2(data_folder3)

plot_boxplots(defect_counts1, defect_counts2, defect_counts3)

#################################################################################
# Bar plot for the imbalanced/balanced data - class count

def plot_defective_non_defective(defective_count, non_defective_count):
    # Data to plot
    labels = ['Defective', 'Non-Defective']
    counts = [defective_count, non_defective_count]
    percentages = [x / sum(counts) * 100 for x in counts]

    # Set font size for all elements
    sns.set(style="whitegrid", context={"font.size":16,"axes.titlesize":16,"axes.labelsize":16}) 

    # Create the bar plot
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=counts, color="skyblue")

    # Add the text with counts and percentages
    for i, (pct, count) in enumerate(zip(percentages, counts)):
        ax.text(i, count, f'{pct:.0f}%\n{int(count)}', ha="center", va="top", color='black', fontsize=16)

    plt.ylabel('Vial count', fontsize=16)
    # plt.xlabel('Category', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

plot_defective_non_defective(defective_count=272, non_defective_count=1875)


