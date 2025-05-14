import os
from pathlib import Path

# Base directories
base_labels_dir = Path("data/carparts-bb/labels")
base_images_dir = Path("data/carparts-bb/images")
splits = ["train", "valid", "test"]

def is_logically_empty(file_path):
    with open(file_path, 'r') as f:
        content = f.read().strip()
    return len(content) == 0

def delete_empty_labels_and_images():
    deleted_count = 0

    for split in splits:
        label_dir = base_labels_dir / split
        image_dir = base_images_dir / split

        for label_file in label_dir.glob("*.txt"):
            if is_logically_empty(label_file):
                image_file = image_dir / f"{label_file.stem}.jpg"

                try:
                    os.remove(label_file)
                    print(f"Deleted empty label: {label_file}")
                    if image_file.exists():
                        os.remove(image_file)
                        print(f"Deleted corresponding image: {image_file}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {label_file} or {image_file}: {e}")

    print(f"\nTotal deleted pairs: {deleted_count}")

if __name__ == "__main__":
    delete_empty_labels_and_images()
