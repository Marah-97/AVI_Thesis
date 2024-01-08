
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from email.mime.text import MIMEText
import io
import sys
from sklearn.linear_model import SGDClassifier

# Building SVM classifier using SGD.
# Loading and preprossecing the images by grouping the vials, showing the results where vials are grouped.
# No need for future extraction as it didnt make difference.

img_size = 256 # image size provided to the SVM

threshold=2 # this threshold is for detemining vial is defect or not. used in aggregate_vial_labels function

loss='hinge' # the loss function in SGD, non-liear

################################################################################################################################################################
################################################################################################################################################################
#### Function Definitions ####

def load_and_preprocess_images_labels(image_dir, label_dir, label_value=1, img_size=(img_size,img_size), augment_defects=False, id_to_images=None):
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

def train_svm(X_train, y_train, random_state=42):  # Function to train an SVM classifier using SGD
    sgd_classifier = SGDClassifier(loss=loss, penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=random_state)
    sgd_classifier.fit(X_train, y_train)
    return sgd_classifier

def evaluate_svm(svm_classifier, X_test, y_test): # Function to evaluate an SVM classifier
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

def visualize_vial_predictions(vial_id, id_to_images, actual_labels, predicted_labels): # Function to visualize predictions for a vial

    images = id_to_images[vial_id]
    n = len(images)
    fig, axes = plt.subplots(nrows=n//5 + int(n%5 > 0), ncols=5, figsize=(15, n//5 * 3))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < n:
            image_2d = images[i].reshape(img_size, img_size)
            ax.imshow(image_2d, cmap='gray')
            ax.set_title(f"Pred: {predicted_labels[i]}, Actual: {actual_labels[i]}")
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def aggregate_vial_labels(grouped_labels, threshold=threshold, label_type="actual"): # Function to aggregate labels by vial, 
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

def evaluate_vial_level_plot1(actual, predicted):
    # Calculating various metrics
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    # Generating the confusion matrix
    conf_matrix = confusion_matrix(actual, predicted, normalize='true')
    labels = ['non-defective', 'defective']  # 0: Good, 1: Defective
    # Printing metrics
    print(f"Vial Accuracy: {accuracy}")
    print(f"Vial Precision: {precision}")
    print(f"Vial Recall: {recall}")
    print(f"Vial F1 Score: {f1}")
    print("Vial Confusion Matrix:")
    # Plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap=sns.cubehelix_palette(as_cmap=True), ax=ax ,linewidth=.5, annot_kws={"size": 18})
    # Setting labels, title, and ticks for the plot
    ax.set_xlabel('Predicted Labels', fontsize=18)
    ax.set_ylabel('True Labels', fontsize=18)
    # ax.set_title('Normalized Vial Confusion Matrix', fontsize=28)
    ax.xaxis.set_ticklabels(labels, fontsize=18)
    ax.yaxis.set_ticklabels(labels, fontsize=18)
    plt.show()    


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

def group_labels_by_vial(id_to_images, labels):
    grouped_labels = {}
    index = 0
    for vial_id, images in id_to_images.items():
        grouped_labels[vial_id] = labels[index:index + len(images)]
        index += len(images)
    return grouped_labels

################################################################################################################################################################
################################################################################################################################################################

#### Load Data (validation and test for test) ####
# data_dir = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050"
data_dir = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3_1050"
# data_dir = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5_1050"

image_train_dir = os.path.join(data_dir, 'images', 'train')
label_train_dir = os.path.join(data_dir, 'labels', 'train')
image_valid_test_dirs = [os.path.join(data_dir, 'images', dtype) for dtype in ['validation', 'test']]
label_valid_test_dirs = [os.path.join(data_dir, 'labels', dtype) for dtype in ['validation', 'test']]


# Load, preprocess images and labels 
id_to_images = {}
X_train, y_train ,defect_vials= load_and_preprocess_images_labels(image_train_dir, label_train_dir, label_value=1, augment_defects=True, id_to_images=id_to_images)
X_test, y_test = [np.concatenate(data) for data in zip(*(load_and_preprocess_images_labels(img_dir, lbl_dir, label_value=1, augment_defects=True, id_to_images=None)[:2] 
                    for img_dir, lbl_dir in zip(image_valid_test_dirs, label_valid_test_dirs)))]

# Group images by vial ID for the combined test set
id_to_images_test = {}
for img_dir, lbl_dir in zip(image_valid_test_dirs, label_valid_test_dirs):
    _, _, defect_vials_temp = load_and_preprocess_images_labels(img_dir, lbl_dir, label_value=1, augment_defects=True, id_to_images=id_to_images_test)

################################################################################################################################################################
# Train and Evaluate the SVM classifier using SGD
sgd_classifier = train_svm(X_train, y_train)

print("Combined Test Set:")
evaluate_svm(sgd_classifier, X_test, y_test) 
################################################################################################################################################################


# Group labels by vial 
predicted_labels = sgd_classifier.predict(X_test)
actual_labels_by_vial = group_labels_by_vial(id_to_images_test, y_test)
predicted_labels_by_vial = group_labels_by_vial(id_to_images_test, predicted_labels)

# Find misclassified vials
actual_defect_predicted_good_vials = find_misclassified_vials(actual_labels_by_vial, predicted_labels_by_vial, True)
actual_good_predicted_defect_vials = find_misclassified_vials(actual_labels_by_vial, predicted_labels_by_vial, False)
print("Vials that are actually defective but predicted as good:", actual_defect_predicted_good_vials)
print("Vials that are actually good but predicted as defective:", actual_good_predicted_defect_vials)

# Aggregate labels at the vial level
vial_level_actual = aggregate_vial_labels(actual_labels_by_vial,  label_type=      "Actual   ")
vial_level_predicted = aggregate_vial_labels(predicted_labels_by_vial, label_type="Predicted")

# Evaluate at the vial level and plot results
evaluate_vial_level(vial_level_actual, vial_level_predicted)
evaluate_vial_level_plot(vial_level_actual, vial_level_predicted)
evaluate_vial_level_plot1(vial_level_actual, vial_level_predicted)


################################################################################################################################################################
################################################################################################################################################################
# Visualize predictions for a specific vial
# cam 5 : ['11467', '39859', '5194', '61022', '63799', '39866', '5198', '5202', '61030', '61031', '64747', '64748']
# cam 4 : ['52753', '52791', '52799'] ,  ['52776', '63234', '5199', '52760'] 
# cam 3 : ['63245'] , ['11466', '52743', '63924', '11468', '11473', '5197', '5198', '52776', '52793', '60991', '64745', '64748']
# '63917', '52734' CAM4 --- '63841' CAM3 --- 
one_specific_vial_id = '63245'  
visualize_vial_predictions(one_specific_vial_id, id_to_images_test, actual_labels_by_vial[one_specific_vial_id], predicted_labels_by_vial[one_specific_vial_id])
###################################################################################################################################################

# Plotting the learning curve for SGDClassifier

from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6)) 
common_params = {
    "X": X_train,
    "y": y_train,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Accuracy",
}

LearningCurveDisplay.from_estimator(sgd_classifier, **common_params, ax=ax)
handles, label = ax.get_legend_handles_labels()
ax.legend(handles[:2], ["Training Score", "Test Score"])
ax.set_title(f"Learning Curve for {sgd_classifier.__class__.__name__}")

plt.show()

######################################################################################################################################

# Look at the coefficients and intercept
coefficients = sgd_classifier.coef_
intercept = sgd_classifier.intercept_
print("Model Coefficients:", coefficients)
print("Model Intercept:", intercept)

# If the coefficients are large and the intercept is not too close to zero,
# it may suggest a clear margin, indicating potential linear separability.

######################################################################################################################################

# Plot the ROC curves

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate decision function scores
y_train_scores = sgd_classifier.decision_function(X_train)
y_test_scores = sgd_classifier.decision_function(X_test)

# Calculate ROC curves and AUC scores
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_scores)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_scores)
auc_train = roc_auc_score(y_train, y_train_scores)
auc_test = roc_auc_score(y_test, y_test_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label=f'AUC TRAIN = {auc_train:.2f}')
plt.plot(fpr_test, tpr_test, color='orange', lw=2, label=f'AUC TEST = {auc_test:.2f}')
plt.plot([0, 1], [0, 1], color='green', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC(ROC curve)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
















