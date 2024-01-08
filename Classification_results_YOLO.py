# This file is to generate the confusion matrix from YOLO 
# Given the predicted bounding box, gt bbox and an iou threshold
# In the first part, it do only consider the images without grouping them into vials
# In the second part, it groups the vials and uses a threshold for min. number of imgs to consider vial defect

import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import recall_score

# image_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3_1050\images\test'
# ground_truth_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3_1050\labels\test'
# predicted_directory = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\detect\exp5\labels"

# image_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\images\test'
# ground_truth_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\labels\test'
# predicted_directory = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\detect\exp6\labels"

image_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5_1050\images\test'
ground_truth_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5_1050\labels\test'
predicted_directory = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\detect\exp7\labels"

#################################################################################
#################################################################################

def iou(box1, box2):
    # calculate the iou of two bounding boxes.
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0


def load_bboxes(file_path):
    #load bounding boxes from a file
    bboxes = []
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append((x_center, y_center, width, height))
    return bboxes

def classify_images_based_on_iou(image_dir, pred_dir, gt_dir, iou_threshold=0.5):
    image_files = sorted(os.listdir(image_dir))
    y_true = []
    y_pred = []

    for img_file in image_files:
        img_name = os.path.splitext(img_file)[0]
        gt_file = os.path.join(gt_dir, img_name + '.txt')
        pred_file = os.path.join(pred_dir, img_name + '.txt')

        gt_bboxes = load_bboxes(gt_file)
        pred_bboxes = load_bboxes(pred_file)

        # classify the image based on iou
        gt_label = 1 if gt_bboxes else 0
        pred_label = 0
        for pred_box in pred_bboxes:
            for gt_box in gt_bboxes:
                if iou(pred_box, gt_box) > iou_threshold:
                    pred_label = 1
                    break
            if pred_label:
                break

        y_true.append(gt_label)
        y_pred.append(pred_label)
    print()
    return y_true, y_pred

def classify_images_with_iou(image_dir, pred_dir, gt_dir, iou_threshold=0.5):
    image_files = sorted(os.listdir(image_dir))
    y_true = []
    y_pred = []

    for img_file in image_files:
        img_name = os.path.splitext(img_file)[0]
        gt_file = os.path.join(gt_dir, img_name + '.txt')
        pred_file = os.path.join(pred_dir, img_name + '.txt')

        gt_bboxes = load_bboxes(gt_file)
        pred_bboxes = load_bboxes(pred_file)

        gt_label = 1 if gt_bboxes else 0
        pred_label = 1 if pred_bboxes else 0

        if gt_bboxes and pred_bboxes:
            # Check IoU for each pair of predicted and ground truth bounding boxes
            matched = any(iou(pred_box, gt_box) > iou_threshold for pred_box in pred_bboxes for gt_box in gt_bboxes)
            pred_label = 1 if matched else 0

        y_true.append(gt_label)
        y_pred.append(pred_label)

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=['Non-Defective', 'Defective'])
    print("Classification Report:\n", class_report)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def calculate_overall_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

iou_threshold = 0.30
print("iou_threshold = ",iou_threshold)
print(image_directory)
y_true, y_pred = classify_images_with_iou(image_directory, predicted_directory, ground_truth_directory, iou_threshold)
plot_confusion_matrix(y_true, y_pred)
calculate_overall_metrics(y_true, y_pred)

#################################################################################
#################################################################################


threshold_def_imgs = 2

print("threshold_def_imgs = ", threshold_def_imgs)
print(image_directory)


def classify_images_and_aggregate(image_dir, pred_dir, gt_dir, threshold_def_imgs=threshold_def_imgs):
    # get a list of all unique vial IDs based on the image names in the directory
    all_images = sorted(os.listdir(image_dir))
    vial_ids = set(extract_vial_id(img_name) for img_name in all_images)

    # initialize dictionaries to count defective images per vial
    vial_defect_count_pred = defaultdict(int)
    vial_defect_count_gt = defaultdict(int)

    # Classify each image based on the predicted and ground truth bounding boxes
    for img_name in all_images:
        vial_id = extract_vial_id(img_name)
        pred_file = os.path.join(pred_dir, os.path.splitext(img_name)[0] + '.txt')
        gt_file = os.path.join(gt_dir, os.path.splitext(img_name)[0] + '.txt')

        # Increase count if a prediction file exists and is not empty (indicating a defect)
        if is_defective(pred_file):
            vial_defect_count_pred[vial_id] += 1
        # Increase count if a ground truth file exists and is not empty
        if os.path.exists(gt_file) and is_defective(gt_file):
            vial_defect_count_gt[vial_id] += 1
        elif not os.path.exists(gt_file):
            # if the GT file does not exist, it is considered non-defective
            vial_defect_count_gt[vial_id] = max(vial_defect_count_gt[vial_id], 0)

    # classify vials based on the number of defective images exceeding the threshold
    y_true, y_pred = [], []
    for vial_id in vial_ids:
        y_pred.append(1 if vial_defect_count_pred[vial_id] >= threshold_def_imgs else 0)
        y_true.append(1 if vial_defect_count_gt[vial_id] >= threshold_def_imgs else 0)

    return y_true, y_pred

def extract_vial_id(img_name):
    # extract the vial ID from the image name
    parts = img_name.split('-')
    if parts and 'vialID' in parts[0]:
        return parts[0].split('vialID')[1]
    return None

def is_defective(file_path):
    # check if the file exists and contains data (non-empty)
    return os.path.exists(file_path) and os.path.getsize(file_path) > 0

def plot_confusion_matrix(y_true, y_pred):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=['Non-Defective', 'Defective'])

    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Non-Defective', 'Predicted Defective'],
                yticklabels=['Actual Non-Defective', 'Actual Defective'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    print("Classification Report:\n", class_report)

def calculate_overall_metrics(y_true, y_pred):
    # calculate the overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # print thm
    print(f"Overall Accuracy: {accuracy:.2f}")
    print(f"Overall Precision: {precision:.2f}")
    print(f"Overall Recall: {recall:.2f}")
    print(f"Overall F1-Score: {f1:.2f}")


y_true, y_pred = classify_images_and_aggregate(image_directory, predicted_directory, ground_truth_directory)
calculate_overall_metrics(y_true, y_pred)
plot_confusion_matrix(y_true, y_pred)
#-------------------------------------------------------------------
# New function to plot recall against different threshold values

def find_best_threshold_for_recall(image_dir, pred_dir, gt_dir, threshold_range):
    recall_scores = []
    thresholds = []

    for threshold in threshold_range:
        y_true, y_pred = classify_images_and_aggregate(image_dir, pred_dir, gt_dir, threshold)
        recall = recall_score(y_true, y_pred)
        recall_scores.append(recall)
        thresholds.append(threshold)

    # plotting recall against thresholds
    plt.figure(figsize=(10, 6))  
    plt.plot(thresholds, recall_scores, marker='o', linestyle='-', color='b')
    # plt.title('Recall vs. Threshold', fontsize=16)
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 1.05) 
    plt.grid(True)
    plt.show()

    # find the best threshold that maximizes recall
    best_threshold = thresholds[np.argmax(recall_scores)]
    print(f"Best threshold for recall is: {best_threshold} with a recall of: {max(recall_scores):.2f}")

    return best_threshold

threshold_range = range(1, 12) 
find_best_threshold_for_recall(image_directory, predicted_directory, ground_truth_directory, threshold_range)

#-------------------------------------------------------------------
# The normalized confusion matrix
def plot_confusion_matrix_2(actual, predicted):
    # calculating various metrics
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    # the confusion matrix
    conf_matrix = confusion_matrix(actual, predicted, normalize='true')
    labels = ['non-defective', 'defective']  # 0: Good, 1: Defective

    # plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax ,linewidth=.5, annot_kws={"size": 18}) #sns.cubehelix_palette(as_cmap=True)
    # set labels and so on
    ax.set_xlabel('Predicted Labels', fontsize=18)
    ax.set_ylabel('True Labels', fontsize=18)
    # ax.set_title('Normalized Vial Confusion Matrix', fontsize=28)
    ax.xaxis.set_ticklabels(labels, fontsize=18)
    ax.yaxis.set_ticklabels(labels, fontsize=18)
    plt.show()    

print("normalized confusion matrix")
plot_confusion_matrix_2(y_true, y_pred)

