# This file is to find the best number of anchor boxes used in YOLO using k means
# The first pare to find the best k
# the second part to plot the clusters and the anchor boxes

import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# folder_path_1 = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3_1050\labels\train"
# folder_path_2 = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3_1050\labels\test"
# folder_path_3 = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3_1050\labels\validation"

# folder_path_1 = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\labels\train"
# folder_path_2 = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\labels\test"
# folder_path_3 = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\labels\validation"

folder_path_1 = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5_1050\labels\train"
folder_path_2 = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5_1050\labels\test"
folder_path_3 = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5_1050\labels\validation"
folder_path = os.path.join(folder_path_1, folder_path_2, folder_path_3)


def load_yolo_bboxes(folder_path):
    bboxes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                for line in file:
                    _, _, _, w, h = map(float, line.split())
                    bboxes.append([w, h])
    return np.array(bboxes)

def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_

def kmeans(bboxes, k):
    kmeans = KMeans(n_clusters=k, random_state=2)
    kmeans.fit(bboxes)
    return kmeans.cluster_centers_

def avg_iou(bboxes, centroids):
    return np.mean([np.max(iou(bbox, centroids)) for bbox in bboxes])

# folder path containing YOLO bounding boxes
# folder_path = r"C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\labels\test"
folder_path = os.path.join(folder_path_1, folder_path_2, folder_path_3)

# Load bounding boxes
bboxes = load_yolo_bboxes(folder_path)

# Calculate iou's for different numbers of clusters
ks = range(1, 11)
ious = []

for k in ks:
    centroids = kmeans(bboxes, k)
    ious.append(avg_iou(bboxes, centroids))

# Plotting the mean IOU vs number of clusters
plt.figure(figsize=(12, 6))
plt.plot(ks, ious, marker='o')
plt.title('The Mean IOU vs Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Mean IOU')
plt.xticks(ks)
plt.grid(True)
plt.show()

#################################################################################
#################################################################################
# Plot the clusters and the anchor boxes

def kmeans_bboxes(bboxes, n_clusters=9):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(bboxes)
    return kmeans

def plot_clusters(bboxes, centroids):
    plt.scatter(bboxes[:, 0], bboxes[:, 1], c=kmeans.labels_, cmap='viridis', s=30)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, marker='x')
    plt.xlabel('Width', fontsize=18)
    plt.ylabel('Height', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def plot_anchor_boxes(centroids):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Width', fontsize=18)
    ax.set_ylabel('Height', fontsize=18)
    for centroid in centroids:
        rect = Rectangle((0.5 - centroid[0] / 2, 0.5 - centroid[1] / 2), centroid[0], centroid[1], 
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


bboxes = load_yolo_bboxes(folder_path)

# Apply k-means clustering
n_clusters = 5 
kmeans = kmeans_bboxes(bboxes, n_clusters=n_clusters)

# Plot clusters
plot_clusters(bboxes, kmeans.cluster_centers_)

# Plot anchor boxes
plot_anchor_boxes(kmeans.cluster_centers_)

# Print anchor box shapes
print("Anchor Box Shapes:")
for i, centroid in enumerate(kmeans.cluster_centers_):
    print(f"Anchor Box {i+1}: Width: {centroid[0]:.4f}, Height: {centroid[1]:.4f}")
