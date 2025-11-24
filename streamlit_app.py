import streamlit as st
import pydicom
import numpy as np
import requests
import json
from PIL import Image
from io import BytesIO
import os

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Resize, ScaleIntensity
)
from monai.data import MetaTensor

# === Title ===
st.title("🩻 MedPrompt: DICOM Analyzer + Note Generator")

# === File Upload ===
uploaded_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])

if uploaded_file:
    st.success("✅ DICOM file read successfully.")

    ds = pydicom.dcmread(uploaded_file)

    st.subheader("📄 DICOM Metadata")
    metadata = {
        "Patient Name": str(ds.get("PatientName", "")),
        "Modality": str(ds.get("Modality", "")),
        "Scan Date": str(ds.get("StudyDate", "")),
    }
    st.json(metadata)

    # === Image Preview ===
    if 'PixelData' in ds:
        image = ds.pixel_array.astype(np.float32)
        if image.ndim == 3:
            image = image[0]
        image = (255 * (image - np.min(image)) / (np.max(image) - np.min(image))).astype(np.uint8)
        img = Image.fromarray(image)
        st.image(img, caption="DICOM Image Preview")
    else:
        st.warning("⚠️ No pixel data found in DICOM file.")

    # === MONAI Transform & Dummy Inference ===
    try:
        st.subheader("🧬 MONAI Inference (Simulated)")

        monai_transform = Compose([
            EnsureChannelFirst(),
            Resize((256, 256)),
            ScaleIntensity()
        ])

        transformed_image = monai_transform(image)
        monai_output = np.random.rand(1, 256, 256)  # Replace with real model output later

        st.info("✅ MONAI preprocessing completed.")
        st.image(monai_output[0], caption="Simulated MONAI Output")
    except Exception as e:
        st.warning(f"⚠️ MONAI inference skipped: {e}")

    # === Clinical Note Generation ===
    st.subheader("🧠 Generate Clinical Note")

    if st.button("📋 Generate Note"):
        # Prepare LLM prompt
        prompt = f"""
You are a helpful radiology assistant. Based on the metadata and image description below, generate a concise clinical note.

Patient Metadata:
- Name: {metadata['Patient Name']}
- Modality: {metadata['Modality']}
- Scan Date: {metadata['Scan Date']}

Output format:
- Summary
- Findings
- Suggested Follow-up
        """

        # === Local LLM API (via LM Studio)
        try:
            response = requests.post(
                "http://localhost:1234/v1/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "prompt": prompt,
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "stop": ["\n\n"]
                }
            )
            note = response.json()["choices"][0]["text"]
            st.text_area("📋 Clinical Note", note, height=200)
        except Exception as e:
            st.error(f"❌ LLM failed: {e}")
