import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import hydra
from lightning import Trainer
from omegaconf import OmegaConf
import torch
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel
from yolo.tools.drawer import draw_bboxes
from yolo.utils.bounding_box_utils import create_converter
from yolo.utils.model_utils import PostProcess

def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess an image for YOLO inference."""
    # Read image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    img = cv2.resize(img, (640, 640))
    
    # Normalize and convert to tensor
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: Config):
    # Set device to MPS if available
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create inference model
    model = InferenceModel(cfg)
    model = model.to(device)
    model.eval()
    
    # Setup post-processing
    vec2box = create_converter(
        cfg.model.name, model.model, cfg.model.anchor, cfg.image_size, device
    )
    post_process = PostProcess(vec2box, cfg.task.nms)
    
    # Get COCO validation images
    current_dir = Path(os.getcwd())
    print(f"Current working directory: {current_dir}")
    
    coco_val_path = current_dir / "data/coco/images/val2017"
    print(f"Looking for COCO validation images at: {coco_val_path}")
    print(f"Directory exists: {coco_val_path.exists()}")
    
    if not coco_val_path.exists():
        print(f"Error: COCO validation directory not found at {coco_val_path}")
        print("Available directories:")
        for root, dirs, files in os.walk(current_dir):
            print(f"  {root}")
        return
    
    # Get first 5 image files
    image_files = list(coco_val_path.glob("*.jpg"))[:5]
    print(f"Processing {len(image_files)} images")
    
    # Process each image
    for image_path in image_files:
        try:
            print(f"\nProcessing image: {image_path}")
            # Load and preprocess image
            image_tensor = load_image(str(image_path)).to(device)
            
            # Run inference and post-process
            with torch.no_grad():
                raw_predictions = model(image_tensor)
                predictions = post_process(raw_predictions, image_size=[640, 640])
            
            # Convert tensor to PIL Image for drawing
            image_pil = Image.fromarray((image_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            
            # Draw bounding boxes
            image_with_boxes = draw_bboxes(image_pil, predictions, idx2label=cfg.dataset.class_list)
            
            # Convert back to numpy array for OpenCV display
            image_np = np.array(image_with_boxes)
            
            # Display the result
            cv2.imshow("YOLO Detection Results", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            print("Press any key to continue to next image...")
            cv2.waitKey(0)
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")
            continue
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()