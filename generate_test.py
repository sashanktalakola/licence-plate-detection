import os

test_images = []
TEST_DIR = os.path.join("data", "test")
for file_name in os.listdir(TEST_DIR):
	if file_name.endswith(".jpg"):
		FILE_PATH = os.path.join(TEST_DIR, file_name)
		test_images.append(FILE_PATH)

TXT_FILE_PATH = os.path.join("data", "test.txt")
with open(TXT_FILE_PATH, "w") as f:
	for line in test_images:
		f.write("{}\n".format(line))