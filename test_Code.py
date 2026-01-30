import xml.etree.ElementTree as ET
import os

def convert_cvat_xml_to_yolo(xml_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for img in root.findall('image'):
        file_name = img.get('name')
        # Standardize filename to match your .jpg images
        base_name = os.path.splitext(file_name)[0]
        width = float(img.get('width'))
        height = float(img.get('height'))

        label_path = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(label_path, 'w') as f:
            for box in img.findall('box'):
                # CVAT XML uses absolute pixels: xtl, ytl, xbr, ybr
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                # YOLO needs normalized center_x, center_y, width, height
                w = xbr - xtl
                h = ybr - ytl
                cx = xtl + (w / 2)
                cy = ytl + (h / 2)

                # Normalize by image dimensions
                f.write(f"0 {cx/width:.6f} {cy/height:.6f} {w/width:.6f} {h/height:.6f}\n")
    
    print(f"✅ Success! Created {len(root.findall('image'))} label files in {output_dir}")

# RUN IT
convert_cvat_xml_to_yolo('annotations.xml', 'labels')