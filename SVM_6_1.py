import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


# Jeg skal lige finde fejlen med det når vi gruppere billederne for en vial pg predicte osv. 
# For jeg får label 0 for en defekt test image, og den burdt være 1 
# Jeg skal også visalisere flere billeder og vials for at sikre mig resultatet
# Jeg skal lege med resize 64,64 måske har vi brug for mere?


# Function to extract Local Binary Pattern (LBP) features from an image
def extract_lbp_image(image, radius=3, n_points=24):
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp_image


def load_and_preprocess_images_labels(image_dir, label_dir, label_value, id_to_images=None):
    image_data = []
    labels = []

    # List all files in the image directory
    image_files = os.listdir(image_dir)

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (64, 64))
        lbp_image = extract_lbp_image(image)
        lbp_flat = lbp_image.flatten()  # Flatten the LBP features


        vial_id = image_file.split('vialID')[1].split('-')[0]

        # Group images by vial ID
        if id_to_images is not None:
            id_to_images.setdefault(vial_id, []).append(lbp_flat)  # Group flattened images by vial ID

        image_data.append(lbp_flat)

        # Check if a corresponding label file exists and is not empty
        label_file_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))
        if os.path.exists(label_file_path) and os.path.getsize(label_file_path) > 0:
            labels.append(label_value)
        else:
            labels.append(0)  # No defect

    return np.array(image_data), np.array(labels)



# Function to train an SVM classifier
def train_svm(X_train, y_train):
    svm_classifier = svm.SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

# Function to evaluate an SVM classifier
def evaluate_svm(svm_classifier, X_test, y_test):
    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)


# Function to make a decision for each vial
def decide_vial_defect(id_to_images, svm_classifier, threshold=2):
    vial_defects = {}
    for vial_id, images in id_to_images.items():
        predictions = svm_classifier.predict(images)
        defect_count = np.sum(predictions)
        vial_defects[vial_id] = defect_count > threshold
    return vial_defects

def visualize_vial_predictions(vial_id, id_to_images, actual_labels, predicted_labels):
    images = id_to_images[vial_id]
    n = len(images)
    fig, axes = plt.subplots(nrows=n//5 + int(n%5 > 0), ncols=5, figsize=(15, n//5 * 3))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            # Reshape the flattened image to 2D
            image_2d = images[i].reshape(64, 64)
            ax.imshow(image_2d, cmap='gray')
            ax.set_title(f"Pred: {predicted_labels[i]}, Actual: {actual_labels[i]}")
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()



# Data directories
data_dir = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\coco128"
image_train_dir = os.path.join(data_dir, 'images', 'train2017')
label_train_dir = os.path.join(data_dir, 'labels', 'train2017')

image_valid_dir = os.path.join(data_dir, 'images', 'valid2017')
label_valid_dir = os.path.join(data_dir, 'labels', 'valid2017')

image_test_dir = os.path.join(data_dir, 'images', 'test2017')
label_test_dir = os.path.join(data_dir, 'labels', 'test2017')




# Load and preprocess images and labels for training, validation, and test
X_train, y_train = load_and_preprocess_images_labels(image_train_dir, label_train_dir, label_value=1)
X_valid, y_valid = load_and_preprocess_images_labels(image_valid_dir, label_valid_dir, label_value=1)
X_test, y_test = load_and_preprocess_images_labels(image_test_dir, label_test_dir, label_value=1)

# Train the SVM classifier
svm_classifier = train_svm(X_train, y_train)

# Evaluate the SVM classifier on the validation set
evaluate_svm(svm_classifier, X_valid, y_valid)



# Group images by vial ID
id_to_images_train = {}
_, _ = load_and_preprocess_images_labels(image_train_dir, label_train_dir, label_value=1, id_to_images=id_to_images_train)
id_to_images_valid = {}
_, _ = load_and_preprocess_images_labels(image_valid_dir, label_valid_dir, label_value=1, id_to_images=id_to_images_valid)
id_to_images_test = {}
_, _ = load_and_preprocess_images_labels(image_test_dir, label_valid_dir, label_value=1, id_to_images=id_to_images_test)



# Decide if each vial is defective
vial_defects_train = decide_vial_defect(id_to_images_train, svm_classifier)
vial_defects_valid = decide_vial_defect(id_to_images_valid, svm_classifier)
vial_defects_test = decide_vial_defect(id_to_images_test, svm_classifier)

# Here `vial_defects_train` will have the decision for each vial in the training dataset

#########################################################################################
#########################################################################################

evaluate_svm(svm_classifier, X_test, y_test)


def create_vial_label_dict(id_to_images, labels):
    label_dict = {}
    for vial_id, images in id_to_images.items():
        label_dict[vial_id] = [labels[i] for i in range(len(images))]
    return label_dict

# Assuming you have a list `y_test` containing the actual labels for the test images
actual_labels_test = create_vial_label_dict(id_to_images_test, y_test)
# Predicting labels for each image in the test set
predicted_labels = svm_classifier.predict(X_test)
# Assuming you have a dictionary `id_to_images_test` that maps vial IDs to test images
predicted_labels_test = create_vial_label_dict(id_to_images_test, predicted_labels)


# Example usage
vial_id = '63908'  # Replace with an actual vial ID from the test set
visualize_vial_predictions(vial_id, id_to_images_test, actual_labels_test[vial_id], predicted_labels_test[vial_id])

