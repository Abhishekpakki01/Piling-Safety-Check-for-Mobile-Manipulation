#!/usr/bin/env python3
"""
System verification script
Tests all dependencies and configuration
"""

print("="*60)
print("Piling Safety BT - System Test")
print("="*60)

tests_passed = 0
tests_total = 6

# Test 1: Python imports
print("\n[1/6] Testing Python imports...")
try:
    import rclpy
    import py_trees
    import ultralytics
    import cv2
    import yaml
    print("✅ All Python packages imported successfully")
    tests_passed += 1
except ImportError as e:
    print(f"❌ Import failed: {e}")

# Test 2: YOLO model
print("\n[2/6] Testing YOLO model...")
try:
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    print("✅ YOLO model loaded")
    tests_passed += 1
except Exception as e:
    print(f"❌ YOLO failed: {e}")

# Test 3: BT nodes import
print("\n[3/6] Testing BT nodes import...")
try:
    from bt_nodes import DetectPile, SetupLocalization
    print("✅ BT nodes imported")
    tests_passed += 1
except Exception as e:
    print(f"❌ BT nodes import failed: {e}")

# Test 4: Config file
print("\n[4/6] Testing config file...")
try:
    import os
    config_path = 'config/spot_config.yaml'
    assert os.path.exists(config_path), f"Config not found: {config_path}"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Config loaded")
    print(f"   Spot IP: {config['spot']['ip']}")
    print(f"   Primary camera: {config['camera']['primary_camera']}")
    tests_passed += 1
except Exception as e:
    print(f"❌ Config failed: {e}")

# Test 5: Spot SDK (optional)
print("\n[5/6] Testing Spot SDK (optional)...")
try:
    import bosdyn.client
    print("✅ Spot SDK available")
    tests_passed += 1
except ImportError:
    print("⚠️  Spot SDK not installed (optional for safety check mode)")

# Test 6: Dataset structure (optional)
print("\n[6/6] Testing dataset structure (optional)...")
import os
dataset_path = os.path.expanduser("~/noodle_dataset")
if os.path.exists(dataset_path):
    train_imgs = len([f for f in os.listdir(f"{dataset_path}/images/train") 
                     if f.endswith(('.jpg', '.png'))]) if os.path.exists(f"{dataset_path}/images/train") else 0
    print(f"✅ Dataset found with {train_imgs} training images")
    if train_imgs >= 50:
        tests_passed += 1
    else:
        print("⚠️  Need at least 50 images for training")
else:
    print("⚠️  Dataset not found (run collect_noodle_images.py)")

# Summary
print("\n" + "="*60)
print(f"RESULT: {tests_passed}/{tests_total} tests passed")
print("="*60)

if tests_passed >= 4:
    print("✅ System ready for basic operation")
    print("\nYou can run:")
    print("  python3 main.py                 # Safety check mode")
    
    if tests_passed >= 5:
        print("  python3 main.py noodle          # Noodle grasping mode")
else:
    print("⚠️  Some critical components missing")
    print("Review errors above and run: pip3 install -r requirements.txt")

print()