import os
import random
from typing import List, Union, Optional
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import to_pil_image

from yolo.tools.drawer import draw_bboxes
from yolo.utils.bounding_box_utils import transform_bbox

# Replace with your actual class list
class_list = [
    "back_bumper", "back_door", "back_glass", "back_left_door", "back_left_light", "back_light",
    "back_right_door", "back_right_light", "front_bumper", "front_door", "front_glass",
    "front_left_door", "front_left_light", "front_light", "front_right_door", "front_right_light",
    "hood", "left_mirror", "object", "right_mirror", "tailgate", "trunk", "wheel"
]

def load_and_draw_images(base_path="/Users/huubal/ScanGo/reykjavik_orientation/yolo_scango/data/carparts-bb", num_images=5):
    subsets = ['train', 'valid', 'test']
    all_images = []

    for subset in subsets:
        image_dir = os.path.join(base_path, subset, 'images')
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        all_images.extend(image_files)

    sampled_images = random.sample(all_images, min(num_images, len(all_images)))

    for img_path in sampled_images:
        label_path = img_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        if not os.path.exists(label_path):
            continue

        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size

        with open(label_path, 'r') as f:
            lines = f.readlines()

        bboxes = []
        for line in lines:
            parts = torch.tensor(list(map(float, line.strip().split())))
            if len(parts) != 5:
                continue  # skip malformed lines

            class_id = parts[0].unsqueeze(0)  # shape: [1]
            bbox = transform_bbox(parts[1:], "xycwh -> xyxy")  # normalized [0–1]

            # Denormalize to pixel coordinates
            bbox[0::2] *= img_width   # x_min, x_max
            bbox[1::2] *= img_height  # y_min, y_max

            bbox = torch.cat((class_id, bbox))  # shape: [5]
            bboxes.append(bbox)

        # Stack all into a single tensor: shape [N, 5]
        if bboxes:
            bboxes = torch.stack(bboxes)

        drawn_img = draw_bboxes(img, [bboxes], idx2label=class_list)

        plt.figure(figsize=(10, 6))
        plt.imshow(drawn_img)
        plt.axis('off')
        plt.title(os.path.basename(img_path))
        plt.show()

# Run the function
load_and_draw_images()
