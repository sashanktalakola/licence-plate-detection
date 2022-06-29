import os, random, shutil


IMAGES_PATH = "./download/vehicle registration plate/images"
ANNOTATIONS_PATH = "./download/vehicle registration plate/darknet"
DEST_IMAGES_PATH = "./data/train"
FULL_IMAGES = os.listdir(IMAGES_PATH)
num_images = len(FULL_IMAGES)
train_ratio = .8 #You can change according to your dataset size
num_train_images = num_images * train_ratio

TRAIN_IMAGES = random.sample(FULL_IMAGES, num_train_images)
for image in TRAIN_IMAGES:
  file_name = image.split(".")[0]
  IMAGE_PATH = os.path.join(IMAGES_PATH, image)
  ANNOTATION_PATH = os.path.join(ANNOTATIONS_PATH, file_name + ".txt")
  shutil.move(IMAGE_PATH, DEST_IMAGES_PATH)
  shutil.move(ANNOTATION_PATH, DEST_IMAGES_PATH)


TEST_IMAGES = os.listdir(IMAGES_PATH)
DEST_IMAGES_PATH = "./data/test"

for image in TEST_IMAGES:
  file_name = image.split(".")[0]
  IMAGE_PATH = os.path.join(IMAGES_PATH, image)
  ANNOTATION_PATH = os.path.join(ANNOTATIONS_PATH, file_name + ".txt")
  shutil.move(IMAGE_PATH, DEST_IMAGES_PATH)
  shutil.move(ANNOTATION_PATH, DEST_IMAGES_PATH)