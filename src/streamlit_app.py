import streamlit as st
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import io

st.set_page_config(page_title="K-Means Image Compressor")

st.title("🎨 K-Means Image Compressor")
st.markdown("""
Upload an image and compress it using **K-Means clustering**.  
- **More clusters** = better quality, less compression  
- **Fewer clusters** = higher compression, less quality  
""")

# Upload the image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Input: Number of clusters
k = st.number_input("Number of Clusters (K)", min_value=2, max_value=128, value=16, step=1)

if uploaded_image and st.button("🚀 Compress Image"):
    # Load original image
    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    compressed_colors = kmeans.cluster_centers_.astype("uint8")
    labels = kmeans.labels_
    compressed_np = compressed_colors[labels].reshape(image_np.shape)
    compressed_image = Image.fromarray(compressed_np)

    # Display both images side-by-side
    st.subheader("📸 Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("**Compressed Image**")
        st.image(compressed_image, use_column_width=True)

    # Save to BytesIO and offer download
    buffer = io.BytesIO()
    compressed_image.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="💾 Download Compressed Image",
        data=buffer,
        file_name="compressed_image.png",
        mime="image/png"
    )
