import streamlit as st
import cv2
import numpy as np
from utils import PencilSketch, CoolingFilter, Cartoonizer, EmbossFilter, SepiaFilter, NegativeFilter, GaussianBlurFilter  # Importing filter classes

# Streamlit page settings
st.title("Interactive Image Filters")
st.write("Select an image and apply different filters to it!")

# Upload image from user
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Convert the uploaded image to the appropriate format
    img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the original image
    st.image(img_rgb, caption="Uploaded Image", use_column_width=True)

    # Choose a filter
    filter_option = st.selectbox("Choose a filter", ["Pencil Sketch", "Cooling Filter", "Cartoonizer", "Sepia", "Negative", "Blur", "Emboss"])

    # Apply the selected filter
    if filter_option == "Pencil Sketch":
        pencil_sketch = PencilSketch()
        img_result = pencil_sketch.render(img_rgb)
        st.image(img_result, caption="Pencil Sketch Filter", use_column_width=True)

    elif filter_option == "Cooling Filter":
        cooling_filter = CoolingFilter()
        img_result = cooling_filter.render(img_rgb)
        st.image(img_result, caption="Cooling Filter", use_column_width=True)

    elif filter_option == "Cartoonizer":
        cartoonizer = Cartoonizer()
        img_result = cartoonizer.render(img_rgb)
        st.image(img_result, caption="Cartoonizer Filter", use_column_width=True)

    elif filter_option == "Sepia":
        sepia_filter = SepiaFilter()
        img_result = sepia_filter.render(img_rgb)
        st.image(img_result, caption="Sepia Filter", use_column_width=True)

    elif filter_option == "Negative":
        negative_filter = NegativeFilter()
        img_result = negative_filter.render(img_rgb)
        st.image(img_result, caption="Negative Filter", use_column_width=True)

    elif filter_option == "Blur":
        gaussian_blur_filter = GaussianBlurFilter()
        img_result = gaussian_blur_filter.render(img_rgb)
        st.image(img_result, caption="Gaussian Blur Filter", use_column_width=True)

    elif filter_option == "Emboss":
        emboss_filter = EmbossFilter()
        img_result = emboss_filter.render(img_rgb)
        st.image(img_result, caption="Emboss Filter", use_column_width=True)
