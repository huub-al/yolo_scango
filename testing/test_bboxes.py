from pathlib import Path

# Paths
base_dir = Path("data/carparts-bb")
image_base = base_dir / "images"
label_base = base_dir / "labels_xyxy"
splits = ["train", "valid", "test"]

def validate_format(line, line_num):
    parts = line.strip().split()
    if len(parts) != 5:
        return f"Line {line_num}: Expected 5 elements, got {len(parts)} -> {line.strip()}"

    try:
        cls = int(parts[0])
        bbox = list(map(float, parts[1:]))
        if cls < 0:
            return f"Line {line_num}: Class ID should be ≥ 0 -> {line.strip()}"
        if not all(0 <= val <= 1 for val in bbox):
            return f"Line {line_num}: Bbox values out of bounds -> {line.strip()}"
    except ValueError:
        return f"Line {line_num}: Non-numeric values -> {line.strip()}"

    return None  # Line is valid

def run_validation():
    issues = []
    total_images = total_labels = 0

    for split in splits:
        img_dir = image_base / split
        lbl_dir = label_base / split

        image_files = sorted(img_dir.glob("*.jpg"))
        label_files = sorted(lbl_dir.glob("*.txt"))

        image_stems = {img.stem for img in image_files}
        # print(image_stems.pop())
        label_stems = {lbl.stem for lbl in label_files}

        # Check for mismatches
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        for stem in missing_labels:
            issues.append(f"[Missing Label] {split}/images/{stem}.jpg → no matching label file.")

        for stem in missing_images:
            issues.append(f"[Missing Image] {split}/labels/{stem}.txt → no matching image file.")

        # Check label file contents
        for label_file in label_files:
            with open(label_file, "r") as f:
                lines = f.readlines()

            if not lines or all(line.strip() == "" for line in lines):
                issues.append(f"[Empty] {split}/labels/{label_file.name} is empty or only whitespace.")
                continue

            for idx, line in enumerate(lines):
                issue = validate_format(line, idx + 1)
                if issue:
                    issues.append(f"[Format Error] {split}/labels/{label_file.name} - {issue}")

        total_images += len(image_files)
        total_labels += len(label_files)

    print(f"\nChecked {total_images} images and {total_labels} label files.")
    if issues:
        print(f"\n⚠️ Found {len(issues)} issues:")
        for issue in issues:
            print("-", issue)
    else:
        print("✅ All files are valid!")

if __name__ == "__main__":
    run_validation()
