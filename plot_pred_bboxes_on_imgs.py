# Uses the iou and NMS to determine the final detection and plot them on the images


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

score_threshold=0.25
iou_threshold=0.45

def iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

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

    # Compute the IoU
    iou = inter_area / union_area
    return iou

def nms(bboxes, iou_threshold, score_threshold):
    """Apply non-maximum suppression to exclude overlapping bboxes."""
    bboxes = sorted(bboxes, key=lambda x: x[5], reverse=True)
    final_bboxes = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or 
                  iou(box[1:5], chosen_box[1:5]) < iou_threshold or
                  box[5] < score_threshold]

        final_bboxes.append(chosen_box)

    return final_bboxes

def get_images_for_vial(image_dir, vial_id, max_images=30):
    """Get a list of image filenames for a specific vial ID."""
    all_images = sorted(os.listdir(image_dir))
    vial_images = [img for img in all_images if f"vialID{vial_id}" in img]
    return vial_images[:max_images]

def plot_yolo_bboxes_in_grid(image_dir, bbox_dir, gt_bbox_dir, vial_id, score_threshold=score_threshold, iou_threshold=iou_threshold, rows=6, cols=5):
    """Plot a grid of all the30 images after applying iou,NMS for a specific vial ID."""
    images = get_images_for_vial(image_dir, vial_id)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 17))  
    axes = axes.flatten()

    for idx, image_name in enumerate(images):
        # Reading ground truth bounding boxes
        gt_bbox_file = os.path.splitext(image_name)[0] + '.txt' 
        gt_bbox_path = os.path.join(gt_bbox_dir, gt_bbox_file)
        gt_box = None
        if os.path.exists(gt_bbox_path):
            with open(gt_bbox_path, 'r') as file:
                line = file.readline()
                parts = line.split()
                if len(parts) == 5: 
                    _, x_center, y_center, width, height = map(float, parts)
                    x = (x_center - width / 2) * img.width
                    y = (y_center - height / 2) * img.height
                    width *= img.width
                    height *= img.height
                    gt_box = (x, y, width, height)

                        
        image_path = os.path.join(image_dir, image_name)

        parts = image_name.split('-')
        vial_id_part = parts[0]  
        pic_index_part = parts[2] 

        # Extracting the actual ID and pic index
        vial_id_extracted = vial_id_part.split('ID')[1]
        pic_index = pic_index_part[3:]
        with Image.open(image_path).convert('L') as img:
            axes[idx].imshow(img, cmap='gray')
            title = f"ID {vial_id_extracted} - pic {pic_index}"
            axes[idx].set_title(title, fontsize=12)
            axes[idx].axis('off')

            bbox_file = os.path.splitext(image_name)[0] + '.txt'
            bbox_path = os.path.join(bbox_dir, bbox_file)

            bboxes = []
            if os.path.exists(bbox_path):
                with open(bbox_path, 'r') as file:
                    for line in file:
                        parts = line.split()
                        if len(parts) == 6:
                            _, x_center, y_center, width, height, score = map(float, parts)
                            x = (x_center - width / 2) * img.width
                            y = (y_center - height / 2) * img.height
                            width *= img.width
                            height *= img.height
                            bboxes.append((0, x, y, width, height, score))

                final_bboxes = nms(bboxes, iou_threshold, score_threshold)
                for _, x, y, width, height, score in final_bboxes:

                    current_iou = iou((x, y, width, height), gt_box) if gt_box else 0
                    # print("current_iou:_________",current_iou)
                    color = 'red' if (score >= score_threshold and current_iou >= iou_threshold) else 'black'
                    alpha = 1.0 if (score >= score_threshold and current_iou >= iou_threshold) else 0.3  # (70% transparent)
                    
                    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor='none', alpha=alpha)
                    axes[idx].add_patch(rect)
                    if (score >= score_threshold and current_iou >= iou_threshold):
                        axes[idx].text(x, y, f"{score:.2f}", color='blue', fontsize=10)

    plt.tight_layout
    plt.show

image_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\images\validation'
real_bbox_dir = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\labels\validation'
bbox_directory = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\val\exp5_val_DATASET_CAM4_1050\labels"

# sanity check
# This is the vial printed in the "C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\train\exp_DATASET_CAM4_1050\val_batch2_pred.jpg"
vial_id = 52742  #52761 #52797   #52737  
plot_yolo_bboxes_in_grid(image_directory, bbox_directory, real_bbox_dir, vial_id)

