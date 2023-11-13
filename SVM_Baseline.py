

## This Code is from Chat GPT (arbejdsmail - teams) som har SVM model der kører
## på 2 folders, og som har feature extraxtion: 

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
 
# Function to extract Local Binary Pattern (LBP) features from an image
def extract_lbp_features(image, radius=3, n_points=24):
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp_image, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)  # Normalize the histogram
    return lbp_hist
 
# Function to load and preprocess images from a directory
def load_and_preprocess_images(directory, label):
    image_data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
            img = cv2.resize(img, (64, 64))  # Resize the image to a common size
            features = extract_lbp_features(img)
            image_data.append(features)
            labels.append(label)
 
    return np.array(image_data), np.array(labels)
 
# Directory paths for good and bad vials
good_vials_dir = r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\Good\KR40433\CAM4"
bad_vials_dir = r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor Thesis\DATA\ChipB\F\CAM4"
 
# Load and preprocess images and labels
good_vials_data, good_vials_labels = load_and_preprocess_images(good_vials_dir, 0)
bad_vials_data, bad_vials_labels = load_and_preprocess_images(bad_vials_dir, 1)
 
# Combine the data and labels
X = np.vstack((good_vials_data, bad_vials_data))
y = np.concatenate((good_vials_labels, bad_vials_labels))
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Create an SVM classifier
svm_classifier = svm.SVC(kernel='linear', C=1.0)
 
# Train the SVM model
svm_classifier.fit(X_train, y_train)
 
# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)
 
# Evaluate the model
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
############################################################################################################################

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
 
# Function to extract Local Binary Pattern (LBP) features from an image
def extract_lbp_features(image, radius=3, n_points=24):
    lbp_image = local_binary_pattern(image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp_image, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)  # Normalize the histogram
    return lbp_hist
 
# Function to load and preprocess images from multiple directories
def load_and_preprocess_images(directories, label):
    image_data = []
    labels = []
 
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
                img = cv2.resize(img, (64, 64))  # Resize the image to a common size
                features = extract_lbp_features(img)
                image_data.append(features)
                labels.append(label)
 
    return np.array(image_data), np.array(labels)
 
# Directories for good and bad vials

good_vials_dirs = [r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Good\KR40424\CAM4",
                   r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Good\KR40433\CAM4",
                   r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Good\KR40597\CAM4",
                   r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Good\KR40655\CAM4",
                   r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Good\KR40664\CAM4"]
bad_vials_dirs = [r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\ChipB\A\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\ChipB\B\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\ChipB\C\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\ChipB\D\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\ChipB\E\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\ChipB\F\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\ChipB\G\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\ChipB\H\CAM4",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Augmented\RandomHorizontalFlip\CAM4_AUG_A",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Augmented\RandomHorizontalFlip\CAM4_AUG_B",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Augmented\RandomHorizontalFlip\CAM4_AUG_C",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Augmented\RandomHorizontalFlip\CAM4_AUG_D",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Augmented\RandomHorizontalFlip\CAM4_AUG_E",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Augmented\RandomHorizontalFlip\CAM4_AUG_F",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Augmented\RandomHorizontalFlip\CAM4_AUG_G",
                  r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Augmented\RandomHorizontalFlip\CAM4_AUG_H"]
 
# Load and preprocess images and labels
good_vials_data, good_vials_labels = load_and_preprocess_images(good_vials_dirs, 0)
bad_vials_data, bad_vials_labels = load_and_preprocess_images(bad_vials_dirs, 1)
 
# Combine the data and labels
X = np.vstack((good_vials_data, bad_vials_data))
y = np.concatenate((good_vials_labels, bad_vials_labels))
 
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Create an SVM classifier
svm_classifier = svm.SVC(kernel='linear', C=1.0)
 
# Train the SVM model
svm_classifier.fit(X_train, y_train)
 
# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)
 
# Evaluate the model
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



