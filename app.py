import streamlit as st
from PIL import Image
import cv2
import numpy as np
from darknet.darknet import *

CFG_PATH = "cfg/yolov4-obj-32.cfg"
TRAIN_DATA_FILE = "train.data"
WEIGHTS_PATH = "backup/exp-32/yolov4-obj_best.weights"


def draw_bounding_box(img, left, top, right, bottom, class_color, label, confidence):
  overlay = img.copy()
  cv2.rectangle(img, (left, top), (right, bottom), color=class_color, thickness=-1)
  alpha = .8
  img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
  cv2.rectangle(img, (left, top), (right, bottom), color=class_color, thickness=1)
  cv2.putText(img, "{} - {:.1f}".format(label, float(confidence)), (left, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.4, class_color, thickness=1)
  #cv2_imshow(image_new)
  return img

if "loaded_network" not in st.session_state:
  network, class_names, class_colors = load_network(CFG_PATH, TRAIN_DATA_FILE, WEIGHTS_PATH)
  st.session_state["loaded_network"] = (network, class_names, class_colors)
else:
  network, class_names, class_colors = st.session_state.loaded_network

width = network_width(network)
height = network_height(network)

def getPrediction(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio


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