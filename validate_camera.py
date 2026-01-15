import cv2
import os
import glob
import time
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
from IPython.display import clear_output

def run_noodle_test(image_folder_path):
    # 1. Initialize YOLO (Logic from validate_camera.py)
    print("⚡ Loading YOLO model...")
    try:
        # We use 'cpu' to ensure stability in all Colab runtimes
        model = YOLO("yolo11n.pt")
        model.to("cpu")
        print("✅ YOLO Model Loaded Successfully.\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 2. Get list of images
    # Supports common formats like .jpg, .jpeg, and .png
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder_path, ext)))
    
    image_files.sort() # Ensure they are in order
    
    if not image_files:
        print(f"❌ No images found in {image_folder_path}. Please check your path.")
        return

    print(f"🚀 Starting test on {len(image_files)} images...")
    time.sleep(2)

    # 3. Process loop
    for i, img_path in enumerate(image_files):
        # Read the frame (Replacement for cap.read())
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue

        # Run YOLO Inference (Exact logic from validate_camera.py)
        # We set verbose=False to keep the output clean
        results = model(frame, verbose=False, device="cpu")
        
        # Draw boxes on the frame
        for r in results:
            annotated_frame = r.plot() 

        # --- Colab Display Logic ---
        # Clear the previous image to simulate a "video feed"
        clear_output(wait=True) 
        
        print(f"Processing Image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        # Replacement for cv2.imshow
        cv2_imshow(annotated_frame) 
        
        # Pause briefly so you can inspect the detection
        time.sleep(0.1) 

    print("\n✅ All images processed!")

# --- EXECUTION ---
# Update this path to wherever you uploaded your 50 images
image_folder = '/content/noodle_images' 
run_noodle_test(image_folder)
