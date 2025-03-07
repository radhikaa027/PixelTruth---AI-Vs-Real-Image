import streamlit as st
import requests
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
from PIL import Image
from io import BytesIO
import os

# Load the trained model
model = load_model("AIGeneratedModelUpdated.h5")

st.set_page_config(
    page_title="AI VS Real Image Detector",
 
    page_icon="🖼",
)

st.title("AI VS Real Image Detector")

# Option to upload an image file
uploaded_file = st.file_uploader("Upload an image file...", type=["jpg", "jpeg", "png"])
# Option to input an image URL
image_url = st.text_input("Enter the URL of the image:", "")

def detect_image_type(image):
    try:
        img_arr = np.array(image)
        # Convert the image to RGB format if it's not already in RGB
        if len(img_arr.shape) == 2:
            img_arr = cv.cvtColor(img_arr, cv.COLOR_GRAY2RGB)
        elif len(img_arr.shape) == 3 and img_arr.shape[2] == 4:
            img_arr = cv.cvtColor(img_arr, cv.COLOR_RGBA2RGB)
        elif len(img_arr.shape) != 3 or img_arr.shape[2] != 3:
            raise ValueError("Unsupported image format")
            
        new_arr = cv.resize(img_arr, (48, 48)) / 255.0
        test = np.array([new_arr])
        result = model.predict(test)
        return "AI Generated" if result[0][0] > 0.5 else "Real"
    except Exception as e:
        st.error(f"Error detecting image type: {e}")
        return None

def fetch_image_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Check if the content type is an image
            content_type = response.headers['content-type']
            if 'image' in content_type:
                return Image.open(BytesIO(response.content))
            else:
                st.error(f"Invalid content type: {content_type}. Please provide a valid image URL.")
                return None
        else:
            st.error(f"Failed to fetch image from URL. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching image from URL: {e}")
        return None

def is_image_url(url):
    _, ext = os.path.splitext(url)
    return ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

if st.button("Detect Image Type"):
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_type = detect_image_type(image)
            st.write(f"The given image is: {image_type}")
            st.image(image, caption='Uploaded Image', use_column_width=True)
        except Exception as e:
            st.error(f"Error processing uploaded image: {e}")
    elif image_url:
        if is_image_url(image_url):
            image = fetch_image_from_url(image_url)
            if image is not None:
                try:
                    image_type = detect_image_type(image)
                    st.write(f"The given image is: {image_type}")
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                except Exception as e:
                    st.error(f"Error detecting image type: {e}")
        else:
            st.error("Invalid image URL. Please provide a valid URL pointing directly to an image file.")
    else:
        st.warning("Please upload an image file or provide the URL of the image.")
