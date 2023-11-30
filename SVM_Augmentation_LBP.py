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


img_size = 64 # image size that goes into SVM
factor_bright = 1.05 # increase 5%
sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])

#### Function Definitions ####

def extract_lbp_image(image, radius=3, n_points=24):
    """Extract Local Binary Pattern features from an image."""
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    return lbp_image

def flip_vertical(image):
    """Apply vertical flipping to the image."""
    return cv2.flip(image, 0)

def flip_horizontal(image):
    """Apply horizontal flipping to the image."""
    return cv2.flip(image, 1)

def increase_brightness(image, factor=factor_bright):
    """Increase the brightness of the image."""
    bright_image = image.astype(np.float64) * factor
    bright_image = np.clip(bright_image, 0, 255)
    return bright_image.astype(np.uint8)

def sharpen_image(image):
    """Apply sharpening to an image to enhance edges."""
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)    
    return sharpened_image


def load_and_preprocess_images_labels(image_dir, label_dir, label_value=1, img_size=(img_size,img_size), augment_defects=False, id_to_images=None):
    """Load, preprocess images and labels, and perform augmentation on defect vials."""
    """ If we plug in the train images folder and the labels, this function will read the images, group them by vialID's,
        then it loop over the vials, if it is defect, it make augmentations for it and give it another vial ID and
        appending their labels, the output will then be for "id_to_images" vialID for all the images we have in the folder with
        the corresponding lbp array, then after that the augmentations for each type"""
    
    image_data, labels = [], []
    defect_vials = set()
    filename_to_label = {}  # Mapping filenames to their labels

    # Load and preprocess all images
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, img_size)

        lbp_image = extract_lbp_image(image)
        lbp_flat = lbp_image.flatten()

        vial_id = image_file.split('vialID')[1].split('-')[0]
        if id_to_images is not None:
            id_to_images.setdefault(vial_id, []).append(lbp_flat)

        image_data.append(lbp_flat)

        label_file_path = os.path.join(label_dir, image_file.replace('.jpg', '.txt'))
        label = label_value if os.path.exists(label_file_path) and os.path.getsize(label_file_path) > 0 else 0
        labels.append(label)
        filename_to_label[image_file] = label
        if label == label_value:
            defect_vials.add(vial_id)

    # Define augmentation functions in a list
    augmentations = [flip_horizontal, increase_brightness, sharpen_image]

    # Apply augmentations if required
    if augment_defects:
        print("Go inside the ----- if augment_defects")
        for augment_index, augment_func in enumerate(augmentations):
            print(augment_index,"Go inside the ----- for enumerate(augmentations)")
            for image_file in os.listdir(image_dir):
                # print(augment_index,"----", image_file,"----","Go inside the ----- for enumerate(augmentations)")
                vial_id = image_file.split('vialID')[1].split('-')[0]

                if vial_id in defect_vials:
                    image_path = os.path.join(image_dir, image_file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, img_size)
                    original_label = filename_to_label[image_file]

                    # Apply augmentation
                    augmented_image = augment_func(image)
                    augmented_lbp = extract_lbp_image(augmented_image).flatten()
                    augmented_vial_id = f"{vial_id}{augment_index}"
                    if id_to_images is not None:
                        id_to_images.setdefault(augmented_vial_id, []).append(augmented_lbp)
                    image_data.append(augmented_lbp)
                    labels.append(original_label)
        print(labels)

    return np.array(image_data), np.array(labels), defect_vials


def train_svm(X_train, y_train): # Function to train an SVM classifier
    svm_classifier = svm.SVC(kernel='linear', C=1.0)
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

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

def aggregate_vial_labels(grouped_labels, threshold=1, label_type="actual"): # Function to aggregate labels by vial, 
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

    print(f"Vial-Level Accuracy: {accuracy}")
    print(f"Vial-Level Precision: {precision}")
    print(f"Vial-Level Recall: {recall}")
    print(f"Vial-Level F1 Score: {f1}")
    print("Vial-Level Confusion Matrix:")
    print(conf_matrix)

def evaluate_vial_level_plot(actual, predicted):
    # Calculating various metrics
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)

    # Generating the confusion matrix
    conf_matrix = confusion_matrix(actual, predicted)
    labels = ['Good', 'Defective']  # Assuming 0: Good, 1: Defective

    # Printing metrics
    print(f"Vial-Level Accuracy: {accuracy}")
    print(f"Vial-Level Precision: {precision}")
    print(f"Vial-Level Recall: {recall}")
    print(f"Vial-Level F1 Score: {f1}")
    print("Vial-Level Confusion Matrix:")

    # Plotting the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)

    # Setting labels, title, and ticks for the plot
    ax.set_xlabel('Predicted Labels', fontsize=14)
    ax.set_ylabel('True Labels', fontsize=14)
    ax.set_title('Vial-Level Confusion Matrix', fontsize=14)
    ax.xaxis.set_ticklabels(labels, fontsize=12)
    ax.yaxis.set_ticklabels(labels, fontsize=12)

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

def visualize_vial_augmentation(id_to_images, defect_vials, img_size=(img_size, img_size)):
    """Visualize original, flipped, and brightened images for defect vials
       This is only to check how the augmentation looks after lbp"""
    for vial_id, images in id_to_images.items():
        # Skip non-defective vials
        if vial_id not in defect_vials:
            continue
        # Check for flipped and brightened counterparts
        flipped_vial_id = f"{vial_id}0"
        brightened_vial_id = f"{vial_id}1"
        sharp_vial_id = f"{vial_id}2"

        flipped_images = id_to_images.get(flipped_vial_id, [])
        brightened_images = id_to_images.get(brightened_vial_id, [])
        sharpen_images = id_to_images.get(sharp_vial_id, [])
        n = len(images)
        fig, axes = plt.subplots(nrows=n, ncols=4, figsize=(15, n * 3))  # 3 columns for original, flipped, brightened

        for i in range(n):
            ax_orig, ax_flip, ax_bright, ax_sharp = axes[i]
            image_orig = images[i].reshape(img_size)
            image_flip = flipped_images[i].reshape(img_size) if i < len(flipped_images) else np.zeros(img_size)
            image_bright = brightened_images[i].reshape(img_size) if i < len(brightened_images) else np.zeros(img_size)
            image_sharp = sharpen_images[i].reshape(img_size) if i < len(sharpen_images) else np.zeros(img_size)

            ax_orig.imshow(image_orig, cmap='gray')
            ax_orig.set_title(f"Original: {vial_id}")
            ax_orig.axis('off')

            ax_flip.imshow(image_flip, cmap='gray')
            ax_flip.set_title(f"Flipped: {flipped_vial_id}")
            ax_flip.axis('off')

            ax_bright.imshow(image_bright, cmap='gray')
            ax_bright.set_title(f"Brightened: {brightened_vial_id}")
            ax_bright.axis('off')

            ax_sharp.imshow(image_sharp, cmap='gray')
            ax_sharp.set_title(f"Sharped: {sharp_vial_id}")
            ax_sharp.axis('off')
        plt.tight_layout()
        plt.show()
        break  # Visualize only the first defective set.

def group_labels_by_vial(id_to_images, labels):
    grouped_labels = {}
    index = 0
    for vial_id, images in id_to_images.items():
        grouped_labels[vial_id] = labels[index:index + len(images)]
        index += len(images)
    return grouped_labels

def send_email(subject, body): # Send me an email when the code is done run 

    sender = "marah.dk@gmail.com"
    receiver = "marah123dk@gmail.com"
    password = "joch ukgd kobz xdcq"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender, password)
    server.send_message(msg)
    server.quit()

#### Load Data (validation and test for test) ####
data_dir = r"C:\Users\marah\OneDrive\Documents\GitHub\Sample_Data"
image_train_dir = os.path.join(data_dir, 'Images', 'train')
label_train_dir = os.path.join(data_dir, 'Labels', 'train')
image_valid_test_dirs = [os.path.join(data_dir, 'Images', dtype) for dtype in ['validation', 'test']]
label_valid_test_dirs = [os.path.join(data_dir, 'Labels', dtype) for dtype in ['validation', 'test']]

# Load, preprocess, and augment images and labels for training
id_to_images = {}
X_train, y_train ,defect_vials= load_and_preprocess_images_labels(image_train_dir, label_train_dir, label_value=1, augment_defects=True, id_to_images=id_to_images)

# only to test if the augmentations works
# augmented_vials = [vial_id for vial_id in id_to_images if vial_id.endswith('0') or vial_id.endswith('1')]
# print(f"Total augmented vials: {len(augmented_vials)}")


# To visualise the augmented types (not a very good function but works)
visualize_vial_augmentation(id_to_images, defect_vials)


# Load, preprocess, and augment images and labels for test 
X_test, y_test = [np.concatenate(data) for data in zip(*(load_and_preprocess_images_labels(img_dir, lbl_dir, label_value=1, augment_defects=True, id_to_images=None)[:2] 
                    for img_dir, lbl_dir in zip(image_valid_test_dirs, label_valid_test_dirs)))]

# Group images by vial ID for the combined test set
id_to_images_test = {}
for img_dir, lbl_dir in zip(image_valid_test_dirs, label_valid_test_dirs):
    _, _, defect_vials_temp = load_and_preprocess_images_labels(img_dir, lbl_dir, label_value=1, augment_defects=True, id_to_images=id_to_images_test)


# Train and Evaluate the SVM classifier
svm_classifier = train_svm(X_train, y_train)
print("Combined Test Set:")
evaluate_svm(svm_classifier, X_test, y_test)


# Group labels by vial 
# to make sure that the train images also were labeled good after augmentation, do same
predicted_labels = svm_classifier.predict(X_test)
actual_labels_by_vial = group_labels_by_vial(id_to_images_test, y_test)
predicted_labels_by_vial = group_labels_by_vial(id_to_images_test, predicted_labels)
print("This is the actual_labels_by_vial \n", actual_labels_by_vial)
print("This is the predicted_labels_by_vial  \n", predicted_labels_by_vial)

actual_labels_by_vial_train = group_labels_by_vial(id_to_images, y_train)
print("This is the actual_labels_by_vial for the train \n", actual_labels_by_vial_train)

# Find misclassified vials
actual_defect_predicted_good_vials = find_misclassified_vials(actual_labels_by_vial, predicted_labels_by_vial, True)
actual_good_predicted_defect_vials = find_misclassified_vials(actual_labels_by_vial, predicted_labels_by_vial, False)
print("Vials that are actually defective but predicted as good:", actual_defect_predicted_good_vials)
print("Vials that are actually good but predicted as defective:", actual_good_predicted_defect_vials)

# Aggregate labels at the vial level
vial_level_actual = aggregate_vial_labels(actual_labels_by_vial, threshold=2, label_type=      "Actual   ")
vial_level_predicted = aggregate_vial_labels(predicted_labels_by_vial, threshold=2, label_type="Predicted")

# Evaluate at the vial level and plot results
evaluate_vial_level_plot(vial_level_actual, vial_level_predicted)

# Visualize predictions for a specific vial
# one_specific_vial_id = '527341'  
# visualize_vial_predictions(one_specific_vial_id, id_to_images_test, actual_labels_by_vial[one_specific_vial_id], predicted_labels_by_vial[one_specific_vial_id])

# Capture the results in a string and send them via email
output = io.StringIO()
sys.stdout = output
evaluate_vial_level(vial_level_actual, vial_level_predicted)
sys.stdout = sys.__stdout__
results_str = output.getvalue()
send_email("Evaluation Results", results_str)


