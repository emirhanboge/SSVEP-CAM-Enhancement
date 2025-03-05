import os
import shutil

base_path = "data/imagenet"
val_images_file = os.path.join(base_path, "val_images.txt")
labels_file = os.path.join(base_path, "imagenet_labels.txt")
val_path = os.path.join(base_path, "val")

with open(labels_file, "r") as f:
    labels = [line.strip() for line in f.readlines()]
label_to_index = {label: idx for idx, label in enumerate(labels)}

with open(val_images_file, "r") as f:
    val_images = [line.strip() for line in f.readlines()]

for label in labels:
    os.makedirs(os.path.join(val_path, str(label_to_index[label])), exist_ok=True)

for line in val_images:
    label, img_name = line.split("/")
    label_index = label_to_index[label]
    src_path = os.path.join(val_path, img_name)
    dest_path = os.path.join(val_path, str(label_index), img_name)

    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"Image {img_name} not found.")
