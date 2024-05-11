import streamlit as st
import numpy as np
import cv2
from keras.models import load_model

# Load your Keras model
model = load_model(".venv/my_model.h5")

def preprocess_data(image):
    image = cv2.resize(image, (50, 50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.reshape(image, (1, 50, 50, 1))
    return image

def main():
    st.title("Image Classifier")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_data(image)

        # Make prediction
        prediction = model.predict(processed_image)
        st.write("Predicted Probabilities:", prediction)

        # Define class labels
        class_labels = ['palm', 'thumb', 'I', 'fist_moved', 'palm', 'fist', 'index', 'fist_moved', 'fist_moved']

        # Get predicted class index
        predicted_class_index = np.argmax(prediction)
        st.write("Predicted Class Index:", predicted_class_index)

        # Display prediction
        st.write("Prediction:", class_labels[predicted_class_index])

if __name__ == "__main__":
    main()
