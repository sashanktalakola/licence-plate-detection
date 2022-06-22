import os

train_images = []
TRAIN_DIR = os.path.join("data", "train")
for file_name in os.listdir(TRAIN_DIR):
	if file_name.endswith(".jpg"):
		FILE_PATH = os.path.join(TRAIN_DIR, file_name)
		train_images.append(FILE_PATH)

TXT_FILE_PATH = os.path.join("data", "train.txt")
with open(TXT_FILE_PATH, "w") as f:
	for line in train_images:
		f.write("{}\n".format(line))