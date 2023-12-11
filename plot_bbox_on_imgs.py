import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def plot_yolo_bboxes(image_dir, bbox_dir, start_idx, end_idx, threshold=0.15):
    """Plot grayscale images with YOLO format bounding boxes and scores."""
    images = sorted(os.listdir(image_dir))[start_idx:end_idx]
    for image_name in images:
        # Read the image
        image_path = os.path.join(image_dir, image_name)
        with Image.open(image_path).convert('L') as img:  # Convert image to grayscale
            fig, ax = plt.subplots(1)
            ax.imshow(img, cmap='gray')  # Display image in grayscale

            # Corresponding bbox file
            bbox_file = os.path.splitext(image_name)[0] + '.txt'
            bbox_path = os.path.join(bbox_dir, bbox_file)

            if os.path.exists(bbox_path):
                with open(bbox_path, 'r') as file:
                    for line in file:
                        parts = line.split()
                        if len(parts) == 6:
                            _, x_center, y_center, width, height, score = map(float, parts)

                            # Convert to pixel values
                            x_center *= img.width
                            y_center *= img.height
                            width *= img.width
                            height *= img.height

                            # Calculate bottom left corner
                            x = x_center - width / 2
                            y = y_center - height / 2

                            # Determine color based on score
                            color = 'red' if score >= threshold else 'black'
                            alpha = 1.0 if score >= threshold else 0.3  # Full opacity for red, 70% transparent for black

                            # Create a Rectangle patch
                            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor='none', alpha=alpha)
                            ax.add_patch(rect)

                            # Print score if greater than threshold
                            if score >= threshold:
                                plt.text(x, y, f"{score:.2f}", color='blue', fontsize=8)
                        else:
                            print(f"Skipping malformed line in file {bbox_file}: {line}")

            plt.title(image_name)
            plt.show()



# Define your directories and range here
image_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_old\images\validation'
bbox_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\val\exp3\labels'

start_index = 6960  # Starting from the first image
end_index = start_index +30   # Plot the first 30 images

plot_yolo_bboxes(image_directory, bbox_directory, start_index, end_index)


##############################################################################################
##############################################################################################
##############################################################################################
# All 30 images in one figure


import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def plot_yolo_bboxes_in_grid(image_dir, bbox_dir, start_idx, end_idx, threshold=0.1, rows=5, cols=6):
    """Plot a grid of grayscale images with YOLO format bounding boxes and scores."""
    images = sorted(os.listdir(image_dir))[start_idx:end_idx]
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))  # Adjust the figsize as needed
    axes = axes.flatten()

    for idx, image_name in enumerate(images):
        # Read the image
        image_path = os.path.join(image_dir, image_name)
        with Image.open(image_path).convert('L') as img:
            axes[idx].imshow(img, cmap='gray')  # Display image in grayscale
            axes[idx].set_title(image_name, fontsize=8)
            axes[idx].axis('off')

            # Corresponding bbox file
            bbox_file = os.path.splitext(image_name)[0] + '.txt'
            bbox_path = os.path.join(bbox_dir, bbox_file)

            if os.path.exists(bbox_path):
                with open(bbox_path, 'r') as file:
                    for line in file:
                        parts = line.split()
                        if len(parts) == 6:
                            _, x_center, y_center, width, height, score = map(float, parts)

                            # Convert to pixel values
                            x_center *= img.width
                            y_center *= img.height
                            width *= img.width
                            height *= img.height

                            # Calculate bottom left corner
                            x = x_center - width / 2
                            y = y_center - height / 2

                            # Determine color and transparency based on score
                            color = 'red' if score >= threshold else 'black'
                            alpha = 1.0 if score >= threshold else 0.3

                            # Create a Rectangle patch
                            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=color, facecolor='none', alpha=alpha)
                            axes[idx].add_patch(rect)

                            # Print score if greater than threshold
                            if score >= threshold:
                                axes[idx].text(x, y, f"{score:.2f}", color='blue', fontsize=6)

    plt.tight_layout()
    plt.show()

# Define your directories and range here
image_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\datasets\DATASET_CAM4_old\images\validation'
bbox_directory = r'C:\Users\marah\OneDrive\Documents\GitHub\yolov5\runs\val\exp3\labels'

start_index = 6960  # Starting from the first image
end_index = start_index +30   # Plot the first 30 images

plot_yolo_bboxes_in_grid(image_directory, bbox_directory, start_index, end_index)
