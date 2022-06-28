import streamlit as st
from PIL import Image
import cv2
import numpy as np
from darknet.darknet import *
from utils import *

CFG_PATH = "cfg/yolov4-obj-32.cfg"
TRAIN_DATA_FILE = "train.data"
WEIGHTS_PATH = "backup/exp-32/yolov4-obj_best.weights"

if "loaded_network" not in st.session_state:
  network, class_names, class_colors = load_network(CFG_PATH, TRAIN_DATA_FILE, WEIGHTS_PATH)
  st.session_state["loaded_network"] = (network, class_names, class_colors)
else:
  network, class_names, class_colors = st.session_state.loaded_network

width = network_width(network)
height = network_height(network)


st.title("Licence Plate Detection")
st.subheader("Input Image")
image = st.file_uploader('')

if image is not None:
	st.image(Image.open(image), width=720)
else:
	st.warning("Please input an image")
	st.stop()

st.subheader("Prediction")
temp = Image.open(image).convert('RGB')
img = np.array(temp)
detections, width_ratio, height_ratio = getPrediction(img, width, height)

for label, confidence, bbox in detections:
  left, top, right, bottom = bbox2points(bbox)
  left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
  img = draw_bounding_box(img, left, top, right, bottom, (0, 255, 0), label, confidence)

st.image(img, width=720)