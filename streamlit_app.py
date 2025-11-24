import streamlit as st
import numpy as np
import pydicom
import base64
from PIL import Image
import io
import requests
import torch

from monai.transforms import (
    Compose, LoadImage, AddChannel, Resize, ScaleIntensity
)
from torchvision.transforms import ToTensor

# --- Streamlit UI ---
st.title("🩻 MedPrompt: DICOM Analyzer + Note Generator")

uploaded_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])
if uploaded_file:
    st.success("✅ DICOM file read successfully.")
    ds = pydicom.dcmread(uploaded_file)
    
    # Metadata
    metadata = {
        "Patient Name": str(getattr(ds, "PatientName", "")),
        "Modality": str(getattr(ds, "Modality", "")),
        "Scan Date": str(getattr(ds, "StudyDate", ""))
    }
    st.subheader("📄 DICOM Metadata")
    st.json(metadata)

    # Pixel Preview
    if "PixelData" in ds:
        try:
            img = ds.pixel_array.astype(np.float32)
            if img.ndim == 3:
                img = img[0]
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
            img = img.astype(np.uint8)
            img_pil = Image.fromarray(img).convert("L")
            st.subheader("🖼️ DICOM Image Preview")
            st.image(img_pil, caption="DICOM Preview", width=300)
        except Exception as e:
            st.warning(f"⚠️ Could not render image: {e}")

    # --- MONAI Inference (Simulated) ---
    st.subheader("🧬 MONAI Inference (Simulated)")

    @st.cache_resource
    def get_monai_transforms():
        return Compose([
            LoadImage(image_only=True),
            AddChannel(),
            Resize((256, 256)),
            ScaleIntensity()
        ])

    transforms = get_monai_transforms()

    try:
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())
        image_tensor = transforms(temp_file)
        st.success("✅ MONAI pipeline applied.")
    except Exception as e:
        st.warning(f"⚠️ MONAI inference skipped: {e}")

    # --- Clinical Note Generator (LLM via LM Studio/Open Source) ---
    st.subheader("🧠 Generate Clinical Note")

    if st.button("🧠 Generate"):
        prompt = f"""Generate a clinical radiology note based on:
Modality: {metadata['Modality']}
Scan Date: {metadata['Scan Date']}
Detected Abnormalities: [Simulated or derived from MONAI here]
Suggest any relevant follow-up steps."""
        
        try:
            response = requests.post(
                "http://localhost:1234/v1/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 512
                },
                timeout=10
            )
            result = response.json()
            note = result.get("choices", [{}])[0].get("text", "").strip()
            st.text_area("📋 Clinical Note", note, height=250)
        except Exception as e:
            st.error(f"❌ LLM failed: {e}")
