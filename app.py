import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
from PIL import ImageOps

# Download the model from Google Drive
url = "https://drive.google.com/uc?id=1ayV0rqOLwAP1MEfSTIR4WuckewbciHD5"
output = "mnist_cnn_model.h5"
gdown.download(url, output, quiet=False)

# Load the model
model = load_model(output)

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
    # Convert the canvas to grayscale and invert colors
    image = Image.fromarray((canvas_result.image_data[:, :, 0] * 255).astype('uint8')).convert("L")
    image = ImageOps.invert(image)  # Invert colors

    # Resize to 28x28 while maintaining aspect ratio
    image = image.resize((28, 28), Image.ANTIALIAS)
    
    # Enhance contrast
    image = ImageOps.autocontrast(image)

    # Normalize pixel values to [0, 1]
    image_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    # Check if the image contains meaningful input (non-zero pixels)
    if np.sum(image_array) > 0:
        st.image(image, caption="Your Digit", width=150)

        # Make prediction
        prediction = model.predict(image_array).argmax()
        st.write(f"### Predicted Digit: {prediction}")
    else:
        st.write("Please draw a digit on the canvas!")



# Footer
st.write("Built with [Streamlit](https://streamlit.io) and TensorFlow")
