import os

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def repair():
    print(f"🔍 Current Script Location: {BASE_DIR}")
    print(f"📂 Folders found here: {os.listdir(BASE_DIR)}")

    # We will try to find the images/labels folders even if they are capitalized
    img_folder = None
    lbl_folder = None

    for folder in os.listdir(BASE_DIR):
        if folder.lower() == 'images': img_folder = os.path.join(BASE_DIR, folder)
        if folder.lower() == 'labels': lbl_folder = os.path.join(BASE_DIR, folder)

    if not img_folder or not lbl_folder:
        print("\n❌ Error: Could not find 'images' and 'labels' folders.")
        print("Please rename your folders to exactly 'images' and 'labels' (all lowercase).")
        return

    # Standardize names to lowercase for the rest of the script
    IMAGE_DIR = img_folder
    LABEL_DIR = lbl_folder

    # 1. STANDARDIZE EXTENSIONS
    for f in os.listdir(IMAGE_DIR):
        if f.lower().endswith(('.jpeg', '.jpg', '.png')):
            old_path = os.path.join(IMAGE_DIR, f)
            # Force everything to .jpg
            new_name = os.path.splitext(f)[0] + '.jpg'
            new_path = os.path.join(IMAGE_DIR, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"🔄 Fixed: {f} -> {new_name}")

    # 2. SYNC
    current_images = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')}
    current_labels = {os.path.splitext(f)[0] for f in os.listdir(LABEL_DIR) if f.endswith('.txt')}

    for img in current_images:
        if img not in current_labels:
            with open(os.path.join(LABEL_DIR, f"{img}.txt"), 'w') as f:
                pass
            print(f"📝 Created empty label for: {img}.jpg")

    print(f"\n✅ SUCCESS! Images: {len(current_images)} | Labels: {len(os.listdir(LABEL_DIR))}")

if __name__ == "__main__":
    repair()