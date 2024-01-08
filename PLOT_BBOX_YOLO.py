# This file is to generate the visual inspection results for YOLO
# both predicted bbox and gt bbox are plotted on the images
# vial grid plot and single image plot
# pred bbox confidence score are derived from the csv file
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd

# image_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3_1050\images\test'
# gt_bbox_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM3_1050\labels\test'
# pred_bbox_directory = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\detect\exp5\labels"
# csv_path = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\detect\exp5\predictions.csv" 

# image_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\images\test'
# gt_bbox_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_1050\labels\test'
# pred_bbox_directory = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\detect\exp6\labels"
# csv_path = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\detect\exp6\predictions.csv" 


image_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5_1050\images\test'
gt_bbox_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM5_1050\labels\test'
pred_bbox_directory = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\detect\exp7\labels"
csv_path = r"C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\detect\exp7\predictions.csv" 

def load_confidence_scores(csv_path):
    # define the column names
    column_names = ['filename', 'classname', 'score']

    # load the csv file
    scores_df = pd.read_csv(csv_path, header=None, names=column_names)

    # create a dictionary mapping filenames to scores
    scores_dict = dict(zip(scores_df['filename'], scores_df['score']))
    return scores_dict


def plot_bboxes(ax, img, bbox_path, is_pred, scores_dict):
    # Helper function to plot bounding boxes and confidence scores
    if os.path.exists(bbox_path) and os.path.getsize(bbox_path) > 0:
        with open(bbox_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # yoloo format: class x_center y_center width height
                    x_center, y_center, width, height = map(float, parts[1:5])

                    x = (x_center - width / 2) * img.width
                    y = (y_center - height / 2) * img.height
                    width *= img.width
                    height *= img.height

                    if is_pred:
                        color = 'red'
                        alpha = 0.60  # 40% transparency for predicted bounding boxes
                        image_name = os.path.basename(bbox_path).replace('.txt', '.jpg')
                        score = scores_dict.get(image_name)
                        if score:
                            ax.text(x, y, f"{score:.2f}", color='blue', fontsize=14, verticalalignment='bottom')
                    else:
                        color = 'green'
                        alpha = 1.0  # for ground truth bounding boxes

                    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor='none', alpha=alpha)
                    ax.add_patch(rect)


def get_images_for_vial(image_dir, vial_id, max_images=30):
    #get a list of image filenames for a specific vial ID
    all_images = sorted(os.listdir(image_dir))
    vial_images = [img for img in all_images if f"vialID{vial_id}" in img]
    return vial_images[:max_images]


def plot_bboxes_in_grid(image_dir, pred_bbox_dir, gt_bbox_dir, vial_id, scores_dict, rows=6, cols=5):
    #plot a grid of grayscale images with both YOLO predicted and ground truth bounding boxes for a specific vial ID
    images = get_images_for_vial(image_dir, vial_id)
    fig, axes = plt.subplots(rows, cols, figsize=(13, 17))  
    axes = axes.flatten()

    for idx, image_name in enumerate(images):
        image_path = os.path.join(image_dir, image_name)
        with Image.open(image_path).convert('L') as img:
            axes[idx].imshow(img, cmap='gray')
            axes[idx].axis('off')

            # Title for each subplot
            parts = image_name.split('-')
            vial_id_extracted = parts[0].split('ID')[1]
            pic_index = parts[2][3:]
            title = f"ID {vial_id_extracted} - Pic {pic_index}"
            axes[idx].set_title(title, fontsize=14)

            # plot gt bbxs
            gt_bbox_file = os.path.splitext(image_name)[0] + '.txt'
            gt_bbox_path = os.path.join(gt_bbox_dir, gt_bbox_file)
            plot_bboxes(axes[idx], img, gt_bbox_path, False, scores_dict)

            # pot predicted bbx with conf
            pred_bbox_file = os.path.splitext(image_name)[0] + '.txt'
            pred_bbox_path = os.path.join(pred_bbox_dir, pred_bbox_file)
            plot_bboxes(axes[idx], img, pred_bbox_path, True, scores_dict)

    plt.tight_layout()
    plt.show()

confidence_scores = load_confidence_scores(csv_path)

vial_id = 63747  
plot_bboxes_in_grid(image_directory, pred_bbox_directory, gt_bbox_directory, vial_id, confidence_scores)


#-------------------------------------------------------------------
# plot single image with both gt and pred bbox

def plot_single_image_with_bboxes(image_dir, pred_bbox_dir, gt_bbox_dir, scores_dict, vial_id, pic_number, figsize=(10, 8)):
    """Plot a single image with both YOLO predicted and ground truth bounding boxes for a specific vial ID and picture number."""
    # Search for the image file based on the vial ID and picture number
    image_file = None
    for file in os.listdir(image_dir):
        if file.startswith(f"crop_vialID{vial_id}-") and f"pic{pic_number:02d}" in file:
            image_file = file
            break

    if not image_file:
        print(f"No image found for Vial ID: {vial_id} and Picture Number: {pic_number}.")
        return

    image_path = os.path.join(image_dir, image_file)

    # Plotting the image
    fig, ax = plt.subplots(figsize=figsize)
    with Image.open(image_path).convert('L') as img:
        ax.imshow(img, cmap='gray')
        # Set title with Vial ID and Picture Number
        ax.set_title(f"ID {vial_id}, pic {pic_number:02d}", fontsize=28)
        ax.axis('off')

        # Plot Ground Truth Bounding Boxes
        gt_bbox_file = image_file.replace('.jpg', '.txt')
        gt_bbox_path = os.path.join(gt_bbox_dir, gt_bbox_file)
        plot_bboxes(ax, img, gt_bbox_path, False, scores_dict)

        # Plot Predicted Bounding Boxes with Confidence Scores
        pred_bbox_file = gt_bbox_file
        pred_bbox_path = os.path.join(pred_bbox_dir, pred_bbox_file)
        plot_bboxes(ax, img, pred_bbox_path, True, scores_dict)

    plt.show()

vial_id = 63747 
pic_number = 25  
plot_single_image_with_bboxes(image_directory, pred_bbox_directory, gt_bbox_directory, confidence_scores, vial_id, pic_number)

####################################################################################################################################################
# get all the unique vial ids from test folder printed in list
def extract_vial_ids(folder_path):
    vial_ids = set()  

    #list all files in the given folder
    for file in os.listdir(folder_path):
        parts = file.split('-')
        if len(parts) > 1 and parts[0].startswith("crop_vialID"):
            vial_id = parts[0].replace("crop_vialID", "")
            vial_ids.add(vial_id)
    return list(vial_ids)

unique_vial_ids = extract_vial_ids(gt_bbox_directory)
print(unique_vial_ids)
