import os
import shutil
import random

IMAGE_DIR = "dataset/images/train"
LABEL_DIR = "dataset/labels/train"

VAL_IMG = "dataset/images/valid"
TEST_IMG = "dataset/images/test"
VAL_LBL = "dataset/labels/valid"
TEST_LBL = "dataset/labels/test"

os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(TEST_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)
os.makedirs(TEST_LBL, exist_ok=True)

images = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]
random.shuffle(images)

val_split = int(0.15 * len(images))
test_split = int(0.25 * len(images))

val_images = images[:val_split]
test_images = images[val_split:test_split]

for img in val_images:
    shutil.move(f"{IMAGE_DIR}/{img}", f"{VAL_IMG}/{img}")
    shutil.move(f"{LABEL_DIR}/{img.rsplit('.',1)[0]}.txt", f"{VAL_LBL}/{img.rsplit('.',1)[0]}.txt")

for img in test_images:
    shutil.move(f"{IMAGE_DIR}/{img}", f"{TEST_IMG}/{img}")
    shutil.move(f"{LABEL_DIR}/{img.rsplit('.',1)[0]}.txt", f"{TEST_LBL}/{img.rsplit('.',1)[0]}.txt")

print("✅ Dataset split completed!")
