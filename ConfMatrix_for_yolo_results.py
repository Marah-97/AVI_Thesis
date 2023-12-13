
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

img_size = 64 
score_threshold=0.25
iou_threshold=0.30
threshold_def_imgs = 1

image_dir = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\images\validation'
real_bbox_dir = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\labels\validation'
pred_bbox_dir = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\val\exp5_val_DATASET_CAM4_1050\labels"



def load_and_preprocess_images_labels(image_dir, label_dir, label_value=1, img_size=(img_size,img_size), id_to_images=None):
    """Load, preprocess images and labels, and perform augmentation on defect vials."""

    image_data, labels = [], []
    defect_vials = set()
    filename_to_label = {}  # Mapping filenames to their labels

    # Load and preprocess all images
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, img_size)

        # lbp_image = extract_lbp_image(image)
        image_flat = image.flatten()

        vial_id = image_file.split('vialID')[1].split('-')[0]
        if id_to_images is not None:
            id_to_images.setdefault(vial_id, []).append(image_flat)

        image_data.append(image_flat)

        label_file_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))
        label = label_value if os.path.exists(label_file_path) and os.path.getsize(label_file_path) > 0 else 0
        labels.append(label)
        filename_to_label[image_file] = label
        if label == label_value:
            defect_vials.add(vial_id)

    return np.array(image_data), np.array(labels), defect_vials

def group_labels_by_vial(id_to_images, labels):
    grouped_labels = {}
    index = 0
    for vial_id, images in id_to_images.items():
        grouped_labels[vial_id] = labels[index:index + len(images)]
        index += len(images)
    return grouped_labels

def parse_yolo_bboxes(file_path, image_size):
    """Parse YOLO format bounding boxes from a file."""
    bboxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 5:
                _, x_center, y_center, width, height = map(float, parts)
                x = (x_center - width / 2) * image_size[0]
                y = (y_center - height / 2) * image_size[1]
                width *= image_size[0]
                height *= image_size[1]
                bboxes.append((x, y, width, height))
    return bboxes

def parse_yolo_bboxes_with_confidence(file_path, image_size):
    """Parse YOLO format bounding boxes with confidence score from a file."""
    bboxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 6:
                _, x_center, y_center, width, height, confidence = map(float, parts)
                x = (x_center - width / 2) * image_size[0]
                y = (y_center - height / 2) * image_size[1]
                width *= image_size[0]
                height *= image_size[1]
                bboxes.append((x, y, width, height, confidence))
    return bboxes

def get_image_size(image_path):
    """Get the size of the image."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)

def iou_nms(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1, con_score = box1  # Predicted bbox with confidence score
    x2, y2, w2, h2 = box2  # Ground truth bbox

    # Calculate the (x, y) coordinates of the intersection rectangle
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Calculate the Union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    if union_area == 0:
        return 0, con_score

    # Compute the IoU
    iou = inter_area / union_area
    return iou, con_score

def find_misclassified_vials(actual_labels, predicted_labels, actual_defect_but_predicted_good=False):
    misclassified_vials = []
    for vial_id, actual in actual_labels.items():
        predicted = predicted_labels[vial_id]
        if actual_defect_but_predicted_good:
            # Looking for vials that are actually defective but predicted as good
            if np.any(actual) and not np.any(predicted):
                misclassified_vials.append(vial_id)
        else:
            # Looking for vials that are actually good but predicted as defective
            if not np.any(actual) and np.any(predicted):
                misclassified_vials.append(vial_id)
    return misclassified_vials

def aggregate_vial_labels(grouped_labels, threshold=threshold_def_imgs, label_type="actual"): # Function to aggregate labels by vial, 
    aggregated_labels = []                                             #used to show the final result for each vial in an array, 
    for vial_id, labels in grouped_labels.items():                     #for both actual and predicted so we can see the differ directly.
        # A vial is defective if it contains a number of defective images greater than or equal to the threshold
        aggregated_labels.append(int(np.sum(labels) >= threshold))
    
    print(f"{label_type}_labels_by_vial with threshold = {threshold}: {aggregated_labels}")
    return aggregated_labels

def evaluate_vial_level(actual, predicted): # Function to evaluate the classifier at the vial level
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    conf_matrix = confusion_matrix(actual, predicted)

    print(f"Vial Accuracy: {accuracy}")
    print(f"Vial Precision: {precision}")
    print(f"Vial Recall: {recall}")
    print(f"Vial F1 Score: {f1}")
    print("Vial Confusion Matrix:")
    print(conf_matrix)

def evaluate_vial_level_plot(actual, predicted):
    # Calculating various metrics
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)

    # Generating the confusion matrix
    conf_matrix = confusion_matrix(actual, predicted)
    labels = ['non-defective', 'defective']  # 0: Good, 1: Defective

    # Printing metrics
    print(f"Vial Accuracy: {accuracy}")
    print(f"Vial Precision: {precision}")
    print(f"Vial Recall: {recall}")
    print(f"Vial F1 Score: {f1}")
    print("Vial Confusion Matrix:")

    # Plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)

    # Setting labels, title, and ticks for the plot
    ax.set_xlabel('Predicted Labels', fontsize=14)
    ax.set_ylabel('True Labels', fontsize=14)
    ax.set_title('Vial Confusion Matrix', fontsize=14)
    ax.xaxis.set_ticklabels(labels, fontsize=12)
    ax.yaxis.set_ticklabels(labels, fontsize=12)

    plt.show()    

def collect_pred_labels_by_vial_corrected(image_dir, pred_bbox_dir, real_bbox_dir, iou_threshold, score_threshold):
    """ This function return a dictionary: keys: vialIds and values: array of image is defect or not """
    pred_labels_by_vial = {}

    # Initialize dictionary with all vial IDs and empty arrays
    all_images = os.listdir(image_dir)
    for image_file in all_images:
        vial_id = image_file.split('vialID')[1].split('-')[0]
        if vial_id not in pred_labels_by_vial:
            vial_images = [img for img in all_images if f"vialID{vial_id}" in img]
            pred_labels_by_vial[vial_id] = [0] * len(vial_images)  # Initialize with zeros

    for image_index, image_file in enumerate(all_images):
        vial_id = image_file.split('vialID')[1].split('-')[0]
        pred_bbox_file_path = os.path.join(pred_bbox_dir, image_file.replace('.jpg', '.txt'))
        real_bbox_file_path = os.path.join(real_bbox_dir, image_file.replace('.jpg', '.txt'))

        if os.path.exists(pred_bbox_file_path):
            image_path = os.path.join(image_dir, image_file)
            image_size = get_image_size(image_path)
            pred_bboxes = parse_yolo_bboxes_with_confidence(pred_bbox_file_path, image_size) if os.path.exists(pred_bbox_file_path) else []
            real_bboxes = parse_yolo_bboxes(real_bbox_file_path, image_size) if os.path.exists(real_bbox_file_path) else []

            for pred_box in pred_bboxes:
                for real_box in real_bboxes:
                    iou, con_score = iou_nms(pred_box, real_box)
                    if iou >= iou_threshold and con_score >= score_threshold:
                        # Find the index of the image in the vial's image list
                        image_list = [img for img in all_images if f"vialID{vial_id}" in img]
                        image_idx = image_list.index(image_file)
                        pred_labels_by_vial[vial_id][image_idx] = 1
                        break

    return pred_labels_by_vial

###################################################################################

predicted_labels_by_vial = collect_pred_labels_by_vial_corrected(image_dir, pred_bbox_dir, real_bbox_dir, iou_threshold, score_threshold)

id_to_images = {}
X_test, y_test ,defect_vials= load_and_preprocess_images_labels(image_dir, real_bbox_dir, label_value=1, id_to_images=id_to_images)


actual_labels_by_vial = group_labels_by_vial(id_to_images, y_test)
# print(actual_labels_by_vial)
print(len(actual_labels_by_vial))
# predicted_labels_by_vial = collect_pred_labels_by_vial_corrected(image_dir, pred_bbox_dir, real_bbox_dir, iou_threshold, score_threshold)
# print(predicted_labels_by_vial)
print(len(predicted_labels_by_vial))



# Find misclassified vials
actual_defect_predicted_good_vials = find_misclassified_vials(actual_labels_by_vial, predicted_labels_by_vial, True)
actual_good_predicted_defect_vials = find_misclassified_vials(actual_labels_by_vial, predicted_labels_by_vial, False)
print("Vials that are actually defective but predicted as good:", actual_defect_predicted_good_vials)
print("Vials that are actually good but predicted as defective:", actual_good_predicted_defect_vials)

# Aggregate labels at the vial level
vial_level_actual = aggregate_vial_labels(actual_labels_by_vial, label_type=      "Actual   ")
vial_level_predicted = aggregate_vial_labels(predicted_labels_by_vial, label_type="Predicted")

# Evaluate at the vial level and plot results
# evaluate_vial_level(vial_level_actual, vial_level_predicted)

evaluate_vial_level_plot(vial_level_actual, vial_level_predicted)















