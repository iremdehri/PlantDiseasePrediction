import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set page configuration for a more vibrant theme
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌱", layout="wide")

# Apply custom CSS styles
st.markdown("""
    <style>
        body {
            background-color: #ff0000 !important;
        }
        .title {
            color: #d5006f;
            font-size: 50px;
            font-weight: bold;
            text-align: center;
            margin-top: 50px;
        }
        .stButton>button {
            background-color: #d5006f;
            color: white;
            font-size: 22px;  /* Increased font size for larger button */
            border-radius: 12px;  /* Slightly bigger border radius */
            padding: 20px 40px;  /* Increased padding for larger button */
        }
        .stButton>button:hover {
            background-color: #ff4081;
        }
        .stImage {
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            max-width: 150px;  /* Set a max width for the image */
            max-height: 150px;  /* Set a max height for the image */
        }
    </style>
""", unsafe_allow_html=True)

# Load model and class names
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.markdown('<p class="title">Plant Disease Classifier</p>', unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        # Display the image with a smaller size
        st.image(image, use_column_width=True)

    with col2:
        if st.button('Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
