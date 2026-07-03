import os
from PIL import Image

# Define paths
yolo_labels_dir = "./assets/AI database/testing/annotations_yolo/"
images_dir = "./assets/AI database/testing/images/"
dota_output_dir = "./assets/AI database/testing/annotations/"

os.makedirs(dota_output_dir, exist_ok=True)

# Map class indices from CVAT to your actual text class names
class_mapping = {
    0: "card"
}

for label_file in os.listdir(yolo_labels_dir):
    if not label_file.endswith('.txt'):
        continue
        
    # Find matching image to extract pixel width/height
    base_name = os.path.splitext(label_file)[0]
    img_path = os.path.join(images_dir, f"{base_name}.png") # Change extension if .png
    
    if not os.path.exists(img_path):
        print(f"Warning: Image not found for {label_file}, skipping.")
        continue
        
    with Image.open(img_path) as img:
        width, height = img.size

    # Convert annotations
    dota_lines = []
    with open(os.path.join(yolo_labels_dir, label_file), 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 9:
                continue
                
            class_idx = int(parts[0])
            class_name = class_mapping.get(class_idx, f"class_{class_idx}")
            
            # De-normalize coordinates back to absolute pixels
            x1 = float(parts[1]) * width
            y1 = float(parts[2]) * height
            x2 = float(parts[3]) * width
            y2 = float(parts[4]) * height
            x3 = float(parts[5]) * width
            y3 = float(parts[6]) * height
            x4 = float(parts[7]) * width
            y4 = float(parts[8]) * height
            
            # DOTA string format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
            dota_line = f"{int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(x3)} {int(y3)} {int(x4)} {int(y4)} {class_name} 0\n"
            dota_lines.append(dota_line)
            
    # Write to new DOTA text file
    with open(os.path.join(dota_output_dir, label_file), 'w') as f:
        f.writelines(dota_lines)

print("Conversion complete!")