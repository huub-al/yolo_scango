import os
import torch

from yolo.utils.bounding_box_utils import transform_bbox


def convert_labels_to_xyxy(label_dir: str, output_dir: str = None):
    if output_dir is None:
        output_dir = label_dir

    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "test", "valid"]:
        split_dir = os.path.join(label_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)

        for file_name in os.listdir(split_dir):
            if not file_name.endswith(".txt"):
                continue

            input_path = os.path.join(split_dir, file_name)
            output_path = os.path.join(output_split_dir, file_name)

            with open(input_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = parts[0]
                bbox = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)

                bbox = transform_bbox(bbox, indicator="xycwh -> xyxy")
                bbox = bbox.clamp(0.0, 1.0)  # Ensure values stay in [0, 1]

                bbox_str = " ".join(f"{x:.6f}" for x in bbox.tolist())
                new_lines.append(f"{class_id} {bbox_str}\n")

            with open(output_path, "w") as f:
                f.writelines(new_lines)

# Example usage
label_root = "data/carparts-bb/labels"
output_root = "data/carparts-bb/labels_xyxy"  # Change to None to overwrite original
convert_labels_to_xyxy(label_root, output_root)
