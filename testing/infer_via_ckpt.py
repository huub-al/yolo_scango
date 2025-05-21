import os
import sys
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import lightning

# === Add YOLOv9 path ===
YOLO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'YOLO'))
sys.path.append(YOLO_PATH)

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import torch



# === Inference on one image ===
weights_path = "/Users/huubal/ScanGo/reykjavik_orientation/yolo_scango/outputs/runs/carparts_v9-t_01/train/carparts_v9-t_01/YOLO/zenn0a6v/checkpoints"
img_path = "/data/carparts-bb/images/train/car4_jpg.rf.5e7803281e9b945e15e2301f3b199739.jpg"

# Load model
device = torch.device("cpu") 
model = lightning.LightningModule.load_from_checkpoint(weights_path, map_location=device)
img_size = 640

# Load image
dataset = LoadImages(img_path, img_size=img_size)
for path, img, im0s, vid_cap, s in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Plot results
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{int(cls)} {conf:.2f}'
                plot_one_box(xyxy, im0s, label=label, color=(0, 255, 0), line_thickness=2)

    # Show result
    plt.imshow(cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB))
    plt.title("Inference Result")
    plt.axis("off")
    plt.show()
    break
