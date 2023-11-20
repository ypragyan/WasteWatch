# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Creating a simple Streamlit app
def main():
    st.title("Waste Classification App")

    # Sidebar with user input
    user_input = st.sidebar.text_input("Enter your name", "John Doe")
    st.sidebar.write(f"Hello, {user_input}!")

    # Main content
    st.header("Waste Image Classification")

    # Upload an image through Streamlit
    uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Waste Image.", use_column_width=True)

        # Convert the image to a format suitable for the model
        img_array = image.img_to_array(image_display)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions using the pre-trained model
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=1)[0][0]

        # Display the waste classification result
        st.subheader("Waste Classification Result:")
        st.write(f"Predicted waste category: {decoded_predictions[1]}")
        st.write(f"Confidence: {decoded_predictions[2]:.2%}")

if __name__ == "__main__":
    main()
