"""
Huub Al

Script to convert segmentation labels to bounding box labels
"""

import os
import glob

# Directories to scan
parent_dir = '/Users/huubal/ScanGo/reykjavik_orientation/yolo_project/data/carparts-seg'
root_dirs = ['train', 'val', 'test']

def polygon_to_bbox(coords):
    xs = coords[::2]
    ys = coords[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return x_center, y_center, width, height

for split in root_dirs:
    label_dir = os.path.join(parent_dir, split, 'labels')
    txt_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    for txt_file in txt_files:
        new_lines = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                class_id = int(parts[0])
                coords = parts[1:]
                x_center, y_center, width, height = polygon_to_bbox(coords)
                new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                new_lines.append(new_line)

        # Overwrite with bounding box version
        with open(txt_file, 'w') as f:
            f.write("\n".join(new_lines) + "\n")
