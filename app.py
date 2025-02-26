import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = tf.keras.models.load_model("models/brain_tumor_model_99.h5")

# Load statistics data
stats_df = pd.read_csv("shap/brain_tumor_dataset.csv")

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']

# Set Streamlit page config
st.set_page_config(page_title="Brain Tumor Diagnosis", layout="wide")

# Custom Styling
page_bg = """
<style>
    body {
        background-image: url('https://source.unsplash.com/1600x900/?medical,hospital');
        background-size: cover;
        color: white;
    }
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
    }
    .sub-title {
        font-size: 20px;
        text-align: center;
    }
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        border-radius: 10px;
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Sidebar for user details
st.sidebar.header("Patient Information")
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
symptoms = st.sidebar.text_area("Symptoms")
image_file = st.sidebar.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

st.markdown("<div class='main-title'>Brain Tumor Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload an MRI image and enter details to get a diagnosis</div>", unsafe_allow_html=True)

if image_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_file, caption="Uploaded MRI Scan", use_column_width=True)
        img = Image.open(image_file)
        img = img.resize((299, 299))  # Ensure correct input size
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)
        
    with col2:
        st.subheader("Diagnosis Result")
        st.write(f"**Predicted Tumor Type:** {predicted_class}")
        st.write(f"**AI Confidence Score:** {confidence}%")
        
        # Fetching statistics from CSV
        tumor_data = stats_df[stats_df['tumor_type'] == predicted_class]
        if not tumor_data.empty:
            st.write(f"**Severity Stage:** {tumor_data['severity_stage'].values[0]}")
            st.write(f"**Recommended Treatment:** {tumor_data['treatment_suggestion'].values[0]}")
            st.write(f"**Survival Rate (5 years):** {tumor_data['survival_rate_5yr'].values[0]}%")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.barplot(x=["Confidence", "Survival Rate"], y=[confidence, tumor_data['survival_rate_5yr'].values[0]], ax=ax, palette="coolwarm")
            ax.set_title("Prediction vs Survival Rate")
            st.pyplot(fig)
        else:
            st.write("No statistics available for this tumor type.")
    
    st.success("Diagnosis Complete! Consult a doctor for further evaluation.")
