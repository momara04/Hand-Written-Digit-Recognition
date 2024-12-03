import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('mnist_cnn_model.h5')

# App title
st.title("Handwritten Digit Recognition")
st.write("Draw a digit (0-9) on the canvas below, and the model will predict it.")

# Create a canvas for user to draw
canvas_result = st_canvas(
    fill_color="#000000",  # Black background
    stroke_width=10,       # Thickness of the digit stroke
    stroke_color="#FFFFFF", # White color for the digit
    background_color="#000000",  # Black background
    width=280,             # Canvas width
    height=280,            # Canvas height
    drawing_mode="freedraw",  # Freehand drawing mode
    key="canvas",
)

# Process the canvas drawing
if canvas_result.image_data is not None:
    # Convert the 280x280 drawing to 28x28 for the model
    image = Image.fromarray((canvas_result.image_data[:, :, 0] * 255).astype('uint8'))  # Extract grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(image_array).argmax()
    st.write(f"### Predicted Digit: {prediction}")

# Footer
st.write("Built with [Streamlit](https://streamlit.io) and TensorFlow")
