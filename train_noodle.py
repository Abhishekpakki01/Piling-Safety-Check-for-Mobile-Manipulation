from ultralytics import YOLO
import os

# 1. Load the pre-trained YOLOv11 Nano model
model = YOLO('yolo11n.pt')

# 2. Run the training
# We use 100 epochs; it might stop early if it stops improving (Patience)
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu',  # Change to 0 if you have an NVIDIA GPU
    name='noodle_v1'
)

print("✅ Training complete!")
print("Your model is saved at: runs/detect/noodle_v1/weights/best.pt")