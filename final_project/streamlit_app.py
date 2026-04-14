import os

import requests
import streamlit as st

API_URL = os.getenv("FINAL_PROJECT_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Final Project - Scene Classifier", layout="wide")
st.title("Real-Estate Scene Classifier")
st.caption("Upload a property photo and get a scene label predicted by the FastAPI model.")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Backend status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.ok:
            payload = health.json()
            st.success(f"API status: {payload.get('status')}")
            st.write(f"Model loaded: {payload.get('model_loaded')}")
            if payload.get("detail"):
                st.warning(payload["detail"])
        else:
            st.error(f"Health check failed: {health.status_code}")
    except Exception as exc:
        st.error(f"Could not reach API at {API_URL}: {exc}")

with col1:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Typical classes: Bedroom, Kitchen, Living room, Street, Coast, etc.",
    )

    if uploaded_file is not None:
        try:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        except TypeError:
            st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

        if st.button("Predict", type="primary"):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }

            with st.spinner("Calling API..."):
                try:
                    response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                    if response.ok:
                        result = response.json()
                        st.success(f"Prediction: {result['label']}")
                        st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

                        st.subheader("Top predictions")
                        for item in result.get("top_k", []):
                            st.write(f"- {item['label']}: {item['probability'] * 100:.2f}%")
                    else:
                        st.error(f"API returned {response.status_code}: {response.text}")
                except Exception as exc:
                    st.error(f"Error connecting to API: {exc}")
