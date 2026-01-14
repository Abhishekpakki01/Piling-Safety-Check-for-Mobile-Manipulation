#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import argparse
from ultralytics import YOLO

def main():
    print("="*60)
    print("🤖 REALSENSE D435i/D456 DIAGNOSTIC TOOL")
    print("="*60)

    # 1. Setup RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device details to confirm which camera is connected
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("❌ FATAL: No RealSense devices detected!")
        print("   Troubleshooting:")
        print("   - Check USB cable (Must be USB 3.0+)")
        print("   - Replug the camera")
        return

    print(f"✅ Found {len(devices)} device(s):")
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"   [{i}] {name} (Serial: {serial})")

    # 2. Configure Stream (Use the first detected camera)
    print("\n[2/4] Configuring streams...")
    # D435i/D456 standard resolution
    width, height = 640, 480 
    fps = 30
    
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        pipeline.start(config)
        print("✅ Pipeline started successfully.")
    except Exception as e:
        print(f"❌ Failed to start pipeline: {e}")
        return

    # 3. Load YOLO Model
    print("\n[3/4] Loading YOLO model...")
    try:
        # Load your noodle model if available, otherwise standard
        try:
            model = YOLO("noodle_training/yolo11n_noodle/weights/best.pt")
            print("📦 Using Custom Noodle Model")
        except:
            model = YOLO("yolo11n.pt")
            print("📦 Using Standard YOLOv11n")
    except Exception as e:
        print(f"⚠️  AI Load Failed: {e}")
        model = None

    # 4. Main Loop
    print("\n[4/4] Starting Video Stream (Press 'q' to quit)...")
    
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run YOLO Inference
            if model:
                results = model(color_image, verbose=False)
                
                for r in results:
                    # Draw standard YOLO boxes
                    color_image = r.plot()
                    
                    # --- EXTRA: Get Depth of Detected Object ---
                    for box in r.boxes:
                        # Get center of the box
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        
                        # Get distance from depth frame at center point
                        # (RealSense gives distance in meters)
                        dist = depth_frame.get_distance(cx, cy)
                        
                        # Draw Distance on screen
                        if dist > 0:
                            label = f"{dist:.2f}m"
                            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
                            cv2.putText(color_image, label, (cx + 10, cy), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show images
            cv2.imshow('RealSense Color + YOLO', color_image)
            
            # Optional: Show depth map (colormapped)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(depth_frame.get_data()), alpha=0.03), cv2.COLORMAP_JET)
            # cv2.imshow('RealSense Depth', depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n✅ Test Complete. Camera released.")

if __name__ == "__main__":
    main()