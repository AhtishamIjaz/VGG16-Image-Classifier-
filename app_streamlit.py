import streamlit as st
import requests
from PIL import Image
import io

# Title of the app
st.title("VGG16 Image Classifier")

st.write("Upload an image and the model will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Button to predict
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Convert image to bytes
        img_bytes = uploaded_file.read()

        # Call FastAPI endpoint
        url = "http://127.0.0.1:8000/predict"  # Make sure FastAPI is running on this URL
        files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}

        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            data = response.json()

            # Display prediction
            st.success(f"Prediction: {data['class_name']}")
            st.info(f"Class ID: {data['class_id']}, Confidence: {data['confidence']:.2f}")

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
