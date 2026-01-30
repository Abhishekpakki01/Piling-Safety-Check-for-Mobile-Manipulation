#!/usr/bin/env python3
import cv2
import time
import argparse

def main():
    print("="*60)
    print("🎥 ROBUST CAMERA TEST (Low Bandwidth Mode)")
    print("="*60)

    # 1. Connect to Camera 0
    cap = cv2.VideoCapture(0)

    # 2. FORCE LOW BANDWIDTH SETTINGS (Critical for WSL)
    # Force MJPG (Compressed)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # Force Low Resolution (640x480 is standard for webcams)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # Force Low FPS
    cap.set(cv2.CAP_PROP_FPS, 15)

    if not cap.isOpened():
        print("❌ Could not open camera. Try --source 1")
        return

    # 3. VERIFY SETTINGS
    # Check what the camera ACTUALLY accepted
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_format = int(cap.get(cv2.CAP_PROP_FOURCC))
    # Convert format to string (e.g., 1196444237 -> 'MJPG')
    fmt_str = "".join([chr((actual_format >> 8 * i) & 0xFF) for i in range(4)])

    print(f"✅ Camera Settings Applied:")
    print(f"   - Resolution: {actual_w:.0f}x{actual_h:.0f}")
    print(f"   - FPS: {actual_fps:.0f}")
    print(f"   - Format: {fmt_str} (Should be MJPG)")

    # 4. START STREAM
    print("\nStarting Stream (Press 'q' to quit)...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame dropped (Timeout).")
                time.sleep(1) # Wait a bit before retrying
                continue

            # Success!
            cv2.putText(frame, f"WSL Camera OK", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Robust Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()