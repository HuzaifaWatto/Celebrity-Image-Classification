import streamlit as st
import requests
from PIL import Image
import io
import pandas as pd

st.title("🎭 Celebrity Classification")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    if st.button("Classify"):
        response = requests.post("http://127.0.0.1:5000/predict", files={"file": img_bytes})

        if response.status_code == 200:
            result = response.json()

            st.subheader(f"🟢 Predicted Celebrity: {result['top_celebrity']}")

            st.write("### Top 5 Similar Celebrities:")
            df = pd.DataFrame(result["similar_celebrities"])
            st.table(df)

        else:
            st.error("❌ Error in classification. Check backend logs.")
