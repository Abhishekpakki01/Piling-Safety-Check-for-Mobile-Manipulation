#!/usr/bin/env python3
import cv2
import time
import argparse
import sys
import numpy as np

def check_yolo_availability():
    """Checks if YOLO is installed and can be loaded."""
    try:
        from ultralytics import YOLO
        # FORCE CPU MODE to avoid GTX 1050 / PyTorch errors
        print("⚡ Loading YOLO model on CPU...")
        model = YOLO("yolo11n.pt")
        model.to("cpu")
        print("✅ YOLO is installed and model loaded (CPU Mode).")
        return model
    except ImportError:
        print("⚠️  ultralytics not installed. Skipping AI test.")
        return None
    except Exception as e:
        print(f"⚠️  YOLO model load failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Camera Health & AI Readiness Check")
    parser.add_argument("--source", type=int, default=0, help="Camera Index (default: 0)")
    parser.add_argument("--no-ai", action="store_true", help="Skip YOLO inference test")
    args = parser.parse_args()

    print("="*60)
    print(f"🎥 CAMERA DIAGNOSTIC TOOL (Source: {args.source})")
    print("="*60)

    # 1. ATTEMPT CONNECTION
    print(f"[1/4] Connecting to camera index {args.source}...")
    cap = cv2.VideoCapture(args.source)

    # --- FIX 1: FORCE MJPG (Fixes WSL 'select timeout' error) ---
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    # ------------------------------------------------------------

    if not cap.isOpened():
        print(f"❌ FATAL: Could not open camera {args.source}.")
        print("   Troubleshooting:")
        print("   - Is it plugged in?")
        print("   - Did you bind/attach it via usbipd?")
        return

    # Get Reported Resolution
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Connection Successful. Native Resolution: {w}x{h}")

    # 2. AI SETUP
    model = None
    if not args.no_ai:
        print("\n[2/4] Checking AI Readiness...")
        model = check_yolo_availability()

    # 3. PERFORMANCE LOOP
    print("\n[3/4] Starting Video Stream (Click window and press 'q' to quit)...")
    
    frame_count = 0
    start_time = time.time()
    fps = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to read frame.")
                break

            # Run YOLO Inference (if enabled)
            if model:
                # --- FIX 2: FORCE CPU INFERENCE (Fixes CUDA error) ---
                results = model(frame, verbose=False, device="cpu")
                
                for r in results:
                    frame = r.plot() # Draw boxes

            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed

            # Overlay Stats
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera Health Check', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n🛑 User pressed 'q'. Exiting...")
                break

    except KeyboardInterrupt:
        print("\n🛑 User pressed Ctrl+C. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("✅ Test Complete.")

if __name__ == "__main__":
    main()