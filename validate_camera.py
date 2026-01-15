import os
import cv2
import numpy as np
from PIL import Image
from base64 import b64decode, b64encode
from google.colab.output import eval_js
from IPython.display import display, Javascript
from ultralytics import YOLO

# 1. LOAD MODEL
# Replace "yolo11n.pt" with the path to your best-trained weights (e.g., "best.pt")
model = YOLO("yolo11n.pt") 

def start_noodle_detection():
    """Starts a live webcam stream with YOLO noodle detection overlay."""
    js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    var labelElement;

    async function initWebcam() {
        div = document.createElement('div');
        video = document.createElement('video');
        video.style.display = 'block';
        video.width = 640;
        video.height = 480;
        
        // Request browser camera permission
        stream = await navigator.mediaDevices.getUserMedia({video: true});
        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        await video.play();

        // Overlay canvas for drawing the noodle bounding boxes
        labelElement = document.createElement('canvas');
        labelElement.width = 640;
        labelElement.height = 480;
        labelElement.style.position = 'absolute';
        labelElement.style.left = video.offsetLeft + 'px';
        labelElement.style.top = video.offsetTop + 'px';
        div.appendChild(labelElement);

        captureCanvas = document.createElement('canvas');
        captureCanvas.width = 640;
        captureCanvas.height = 480;
    }

    async function streamFrame() {
        const ctx = captureCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0, 640, 480);
        const imgData = captureCanvas.toDataURL('image/jpeg', 0.8);
        
        // Pass the frame to the Python function for YOLO inference
        const response = await google.colab.kernel.invokeFunction(
            'notebook.detect_noodle', [imgData], {});
        
        // Clear and draw new boxes on the overlay
        const overlayCtx = labelElement.getContext('2d');
        overlayCtx.clearRect(0, 0, 640, 480);
        
        response.data.boxes.forEach(box => {
            overlayCtx.strokeStyle = "#00FF00"; // Green Box
            overlayCtx.lineWidth = 3;
            overlayCtx.strokeRect(box.x, box.y, box.w, box.h);
            overlayCtx.fillStyle = "#00FF00";
            overlayCtx.fillText(box.label, box.x, box.y > 20 ? box.y - 5 : 10);
        });
        
        requestAnimationFrame(streamFrame);
    }

    initWebcam().then(streamFrame);
    ''')
    display(js)

def detect_noodle_callback(img_b64):
    """Python function that processes a single frame and returns detections."""
    # Decode the base64 image from JavaScript
    _, encoded = img_b64.split(",")
    data = b64decode(encoded)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO Inference (optimized for CPU if necessary)
    results = model(img, verbose=False)
    
    found_boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]} {conf:.2f}"
            
            found_boxes.append({
                'x': x1, 'y': y1, 
                'w': x2 - x1, 'h': y2 - y1,
                'label': label
            })

    return {'boxes': found_boxes}

# Register the callback so JavaScript can 'talk' to Python
import google.colab.kernel
google.colab.kernel.register_callback('notebook.detect_noodle', detect_noodle_callback)

# Start the application
print("🚀 Initializing Live Noodle Detection...")
start_noodle_detection()
