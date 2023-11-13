import json
import os
import cv2
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

# Path to your COCO dataset
coco_annotation_file = r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Annotations\ChipB_annotations\CAM4_F.json" #'path/to/your/annotations/instances.json'
coco_dataset_dir = r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\ChipB\F\CAM4" #'path/to/your/dataset/'
output_dir = r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Augmented\RandomHorizontalFlip\CAM4_AUG_F" #'path/to/save/augmented_data/'
new_annotation_file = r"C:\Users\marah\OneDrive\Skrivebord\Maroooh\KID\Bachelor_Thesis\DATA\Annotations\Augmented_ChipB_annotations\CAM4_aug_F.json" #'path/to/save/updated_annotations.json'

os.makedirs(output_dir, exist_ok=True)

coco = COCO(coco_annotation_file)

# Define augmentation transforms for grayscale images
# Flip all images horizontally
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0), 
    transforms.ToTensor()
])

for img_id in coco.getImgIds():
    img_info = coco.loadImgs(img_id)[0]
    img_info['file_name'] = os.path.basename(img_info['file_name'])
    #print("img_info:", img_info)
    img_path = os.path.join(coco_dataset_dir, img_info['file_name'])

#     img_info = coco.loadImgs(img_id)[0]
#     print("img_info:",img_info)
#     img_path = os.path.join(coco_dataset_dir, img_info['file_name'])
    image_width = img_info['width']
    
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    for ann in annotations:
        bbox = ann['bbox']
        x_min = bbox[0]
        width = bbox[2]

        # Calculate new x_min after horizontal flip
        new_x_min = image_width - x_min - width
        bbox[0] = new_x_min

        # Update the annotation
        ann['bbox'] = bbox



    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = Image.fromarray(img)

    augmented_img = augmentation_transforms(img)

    # Convert tensor to numpy array for saving
    augmented_img_np = augmented_img.mul(255).byte().numpy()
    augmented_img_np = np.transpose(augmented_img_np, (1, 2, 0))
    if augmented_img_np.shape[2] == 1:
        augmented_img_np = augmented_img_np.squeeze(2)


    # To save the augmented image, the file name is still the same adding 1 to the end?
    file_name, file_extension = os.path.splitext(img_info["file_name"])
    augmented_img_filename = f'{file_name}1{file_extension}'

    #augmented_img_filename = f'{img_info["file_name"]}'
    cv2.imwrite(os.path.join(output_dir, augmented_img_filename), augmented_img_np)

with open(new_annotation_file, 'w') as f:
    json.dump(coco.dataset, f)

print("Updated annotations saved to:", new_annotation_file)