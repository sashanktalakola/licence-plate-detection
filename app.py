import streamlit as st
from PIL import Image
import cv2


st.title("Licence Plate Detection")
st.subheader("Input Image")
image = st.file_uploader('')

if image is not None:
	st.image(Image.open(image), width=720)
else:
	st.warning("Please input an image")
	#st.stop()

st.subheader("Prediction")