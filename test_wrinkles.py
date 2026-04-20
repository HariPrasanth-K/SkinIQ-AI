import os
import torch
import supervision as sv
import numpy as np
from PIL import Image
from rfdetr import RFDETRSegSmall
import argparse

# ---------------- CONFIG ---------------- #
CHECKPOINT_PATH = "wrinkles_pig.pth"
INPUT_FOLDER = "test_wrinkles"
OUTPUT_FOLDER = "output_wrinkles"

CLASS_NAMES = ["Pigment", "Wrinkles"]  # 2 classes in checkpoint
NUM_CLASSES = 2
CONF_THRESHOLD = 0.3
# ---------------------------------------- #

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------- LOAD MODEL ---------------- #
model = RFDETRSegSmall(
    num_classes=NUM_CLASSES,
    pretrain_weights=None
)

# Allow unpickling of argparse.Namespace for your checkpoint
torch.serialization.add_safe_globals([argparse.Namespace])

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False  # must be False because checkpoint contains more than just weights
)

model.model.model.load_state_dict(checkpoint["model"], strict=True)
model.model.model.to(device)
model.model.model.eval()
model.optimize_for_inference()  # speed up inference

print("✅ Model loaded successfully")

# ---------------- ANNOTATORS ---------------- #
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

# ---------------- INFERENCE LOOP ---------------- #
for file_name in os.listdir(INPUT_FOLDER):

    if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(INPUT_FOLDER, file_name)
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    with torch.no_grad():
        detections = model.predict(image, threshold=CONF_THRESHOLD)

    # ---------------- HANDLE DETECTIONS ---------------- #
    if len(detections) == 0:
        # No detections → save original image
        print(f"⚠️ No detections in {file_name}")
        annotated = image_np.copy()
    else:
        # Generate labels safely
        labels = [
            f"{CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'class_{class_id}'} {conf:.2f}"
            for class_id, conf in zip(detections.class_id, detections.confidence)
        ]
        # Annotate masks and labels
        annotated = mask_annotator.annotate(scene=image_np.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    # ---------------- SAVE OUTPUT ---------------- #
    annotated = Image.fromarray(annotated)
    output_path = os.path.join(OUTPUT_FOLDER, file_name)
    annotated.save(output_path)

    print(f"✅ Processed {file_name}")

print("🎯 All images processed!")