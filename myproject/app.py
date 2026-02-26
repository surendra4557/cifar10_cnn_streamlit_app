import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Prediction"]
)

# Page Config
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="wide")
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}
.stButton>button {
    background-color: #6C63FF;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.fade-in {
    animation: fadeIn 2s ease-in;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

if page == "Home":

 #Hero section
    st.markdown("<h1 style='text-align:center; color:green;'>🎯 CIFAR-10 Image Classification using CNN</h1>", unsafe_allow_html=True)
    st.write("---")

    st.title("🖼️ CIFAR-10 Image Classification App")
    st.write("Upload an image and the model will predict the class.")
    

    #model information Box
    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset", "CIFAR-10")
    col2.metric("Classes", "10")
    col3.metric("Framework", "TensorFlow")

    st.write("---")

    #class labels
    st.subheader("Model Classes")

    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat',
               'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    cols = st.columns(5)
    for i in range(5):
        cols[i].write(class_names[i])

    for i in range(5, 10):
        cols[i-5].write(class_names[i])

    st.write("---")


# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cifar10_cnn_model (3).h5")
    return model

model = load_model()

# CIFAR-10 Classes
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat',
               'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
      



if page == "Prediction":
    st.subheader("Image of any Model Classes for Prediction")
   # Image Upload
    st.title("Upload Images")
    uploaded_files = st.file_uploader("Predicting One or more than one Images in just single click",type=["jpg", "png", "jpeg"],
 accept_multiple_files=True)
st.warning("⚠ This model is trained on CIFAR-10 dataset (32x32 images). High resolution Google images may give inaccurate results.")
    if uploaded_files:
        for uploaded_file in uploaded_files:
          image = Image.open(uploaded_file)
          st.image(image, caption=uploaded_file.name)


        # Preprocessing
          image = image.convert("RGB")
          image = image.resize((32, 32))   # CIFAR-10 size
          image = np.array(image)
          image = image / 255.0            # Normalization
          image = np.expand_dims(image, axis=0)

          # Prediction
          prediction = model.predict(image)
          predicted_class = class_names[np.argmax(prediction)]
          confidence = np.max(prediction) * 100

          col1,col2 = st.columns([1,1])
          with col1:
              st.image(image,caption=uploaded_file.name)
              st.metric("Model Confidence",f"{confidence:.2f}%")
              # Probability Graph
              prob_df = pd.DataFrame( prediction[0],index=class_names,
columns=["Probability"])

              st.bar_chart(prob_df)   
              
          with col2:
              fig = go.Figure(go.Indicator(
                 mode="gauge+number",
                 value=confidence,
                 title={'text': "Model Confidence"},
                 gauge={
                     'axis': {'range': [0, 100]},
                     'bar': {'color': "blue"},
                     'steps': [
                         {'range': [0, 50], 'color': "lightgray"},
                         {'range': [50, 80], 'color': "gray"},
                         {'range': [80, 100], 'color': "green"},
        ],
    }
))

              st.plotly_chart(fig, use_container_width=True)
              st.subheader(f"Prediction Result is:{predicted_class}")
              st.write(f"Confidence:,{confidence:.2f}%")
 
              st.success("model loaded successfully!")
              st.balloons()
              st.spinner("predicting....")
              st.progress(0)
          st.markdown("""<div style='text-align:center;'>\
<span class='star'>⭐ ⭐ ⭐</span> \
<span class='cracker'>🎆 🎇</span>\
</div>""", unsafe_allow_html=True)
     
                
st.markdown("<center>© 2026 Surendra AI Project | Made with SS</center>", unsafe_allow_html=True)
