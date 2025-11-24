import streamlit as st
import pydicom
import numpy as np
from PIL import Image
import json
import requests
from monai.transforms import (
    Compose, LoadImage, AddChannel, Resize, ScaleIntensity, ToTensor
)

# Title
st.title("🩻 MedPrompt: DICOM Analyzer + Note Generator")

# File uploader
uploaded_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])

if uploaded_file:
    st.success("✅ DICOM file read successfully.")

    # Load DICOM
    dicom = pydicom.dcmread(uploaded_file)

    # Metadata
    st.subheader("📄 DICOM Metadata")
    metadata = {
        "Patient Name": str(dicom.get("PatientName", "")),
        "Modality": str(dicom.get("Modality", "")),
        "Scan Date": str(dicom.get("StudyDate", "")),
    }
    st.json(metadata)

    # Image preview
    st.subheader("🖼️ DICOM Image Preview")
    try:
        img_array = dicom.pixel_array
        img = Image.fromarray(img_array).convert("L")
        st.image(img, caption="DICOM Image", use_column_width=True)
    except Exception as e:
        st.warning(f"Could not display image: {e}")

    # MONAI pipeline (simulated)
    st.subheader("🧬 MONAI Inference (Simulated)")
    try:
        transforms = Compose([
            AddChannel(),
            Resize((128, 128)),
            ScaleIntensity(),
            ToTensor()
        ])
        simulated_output = transforms(img_array)
        st.success("Simulated MONAI transform succeeded.")
    except Exception as e:
        st.error(f"MONAI inference skipped: {e}")

    # LLM Clinical Note Generator
    st.subheader("🧠 Generate Clinical Note")

    if st.button("Generate Note"):
        try:
            prompt = f"Patient CT scan metadata:\n{json.dumps(metadata, indent=2)}"
            payload = {
                "model": "openai/gpt-oss-20b",
                "messages": [
                    {"role": "system", "content": "You are a radiology assistant that writes clinical notes."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 512
            }

            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            result = response.json()
            note = result["choices"][0]["message"]["content"]
            st.text_area("📋 Clinical Note", value=note, height=200)

        except Exception as e:
            st.error(f"❌ LLM failed: {e}")
