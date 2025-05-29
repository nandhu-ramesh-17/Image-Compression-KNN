import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

st.title("K-Means Image Compressor")
st.write("Upload an image and choose number of clusters to compress it.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Cluster count
k = st.slider("Number of Clusters (K)", min_value=2, max_value=124, value=16)

# Process and display
if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype("uint8")
    labels = kmeans.labels_

    compressed_image = centers[labels].reshape(image_np.shape)
    compressed_pil = Image.fromarray(compressed_image)

    st.image(compressed_pil, caption="Compressed Image", use_column_width=True)