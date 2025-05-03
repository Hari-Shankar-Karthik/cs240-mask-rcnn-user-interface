from pycocotools.coco import COCO
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set paths
ann_file = './instances_train2017.json'
img_dir = '../'  # wherever your images are stored

# Load COCO annotations
coco = COCO(ann_file)

# List of image file names
filenames = [
    '000000000036.jpg', '000000000061.jpg', '000000000077.jpg',
    '000000000143.jpg', '000000000315.jpg', '000000000438.jpg',
    '000000000472.jpg', '000000000486.jpg', '000000000575.jpg'
]

for fname in filenames:
    img_id = int(fname.split('.')[0])
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    for idx, ann in enumerate(anns):
        mask = coco.annToMask(ann) * 255  # Convert to binary mask
        mask_img = Image.fromarray(mask.astype(np.uint8))
        
        # Save mask
        out_name = f"{img_id}_mask_{idx}.png"
        mask_img.save(out_name)
