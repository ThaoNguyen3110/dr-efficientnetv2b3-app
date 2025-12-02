import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

REPO_ID = "LilyThao/thao_efficientnetv2b3-dr"
FILENAME_IN_REPO = "best_efficientnetv2b3.keras"
MODEL_PATH = "best_efficientnetv2b3.keras"
IMG_SIZE = (224, 224)

# Define a mapping from numerical labels to human-readable class names
ID2LABEL = {
    0: "0 - No DR",
    1: "1 - Mild",
    2: "2 - Moderate",
    3: "3 - Severe",
    4: "4 - Proliferative DR"
}

# Use Streamlit's cache_resource decorator to load the model only once
@st.cache_resource
def load_model():
    # Load the pre-trained Keras model
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load the model
model = load_model()

# Function to preprocess the uploaded image
def preprocess_image(img: Image.Image):
    img = img.convert("RGB") # Convert image to RGB format
    img = img.resize(IMG_SIZE) # Resize image to the target dimensions
    img_array = np.array(img).astype("float32") # Convert PIL Image to NumPy array and cast to float32
    img_array = np.expand_dims(img_array, axis=0) # Add a batch dimension (e.g., (1, 224, 224, 3))
    return img_array

# Set Streamlit page configuration
st.set_page_config(page_title="DR Classification - EfficientNetV2B3", layout="centered")

# Display the main title of the application
st.title("Diabetic retinopathy severity classification EfficientNetV2B3")

# Display the model name being used
st.write("Model: `best_efficientnetv2b3`")

# Provide a brief description of the application
st.markdown(
    "This application classifies the severity of diabetic retinopathy from retinal images using a fine-tuned EfficientNetV2B3 model."
)

# Create a file uploader widget for image upload
uploaded_file = st.file_uploader(
    "Upload a retinal image", # Label for the uploader
    type=["jpg", "jpeg", "png"] # Accepted file types
)

# Check if a file has been uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file) # Open the uploaded image using PIL
    st.image(image, caption="Image is uploaded", use_column_width=True) # Display the uploaded image

    # Create a button for prediction
    if st.button("Prediction"):
        # Preprocess the uploaded image
        x = preprocess_image(image)

        # Get predictions from the model
        preds = model.predict(x)
        pred_probs = preds[0] # Get probabilities for the single input image

        # Determine the predicted class and its confidence score
        pred_class = int(np.argmax(pred_probs)) # Get the index of the class with highest probability
        pred_conf = float(np.max(pred_probs)) # Get the maximum probability (confidence)

        # Display the predicted result
        st.subheader("Predicted Result")
        st.write(f"Predicted Label {ID2LABEL.get(pred_class, str(pred_class))}") # Display the human-readable label
        st.write(f"Probability {pred_conf:.2%}") # Display the confidence score as a percentage

        # Display probabilities for all classes in a table
        st.markdown("Prob for each class")
        prob_table = {
            "Class": [],
            "Class name": [],
            "Prob": []
        }
        num_classes = len(pred_probs)
        for i in range(num_classes):
            prob_table["Class"].append(i)
            prob_table["Class name"].append(ID2LABEL.get(i, f"class {i}"))
            prob_table["Prob"].append(float(pred_probs[i]))

        st.table(prob_table) # Display the table of class probabilities
