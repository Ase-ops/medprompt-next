import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import base64
import io
import requests

from monai.transforms import Compose, LoadImage, AddChannel, Resize, ScaleIntensity, ToTensor

st.set_page_config(page_title="MedPrompt: DICOM Analyzer + Note Generator")

st.title("🩻 MedPrompt: DICOM Analyzer + Note Generator")

uploaded_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])

if uploaded_file:
    try:
        dcm = pydicom.dcmread(uploaded_file)
        st.success("✅ DICOM file read successfully.")

        # Extract metadata
        patient_name = str(getattr(dcm, "PatientName", ""))
        modality = str(getattr(dcm, "Modality", "Unknown"))
        scan_date = str(getattr(dcm, "StudyDate", getattr(dcm, "AcquisitionDate", "")))

        st.subheader("📄 DICOM Metadata")
        st.json({
            "Patient Name": patient_name,
            "Modality": modality,
            "Scan Date": scan_date
        })

        # Convert to image (if possible)
        if hasattr(dcm, "pixel_array"):
            img_array = dcm.pixel_array
            if len(img_array.shape) == 3:
                img_array = img_array[0]  # first slice

            img = Image.fromarray(np.uint8(img_array / np.max(img_array) * 255))
            st.subheader("🖼️ DICOM Image Preview")
            st.image(img, caption="DICOM Image", use_column_width=True)
        else:
            st.warning("No pixel data found.")

        # MONAI Inference
        st.subheader("🧬 MONAI Inference (Simulated)")
        try:
            monai_transforms = Compose([
                AddChannel(),
                Resize((256, 256)),
                ScaleIntensity(),
                ToTensor()
            ])
            _ = monai_transforms(img_array)
            st.success("✅ MONAI transform applied (replace with real model inference later).")
        except Exception as e:
            st.error(f"⚠️ MONAI inference skipped: {e}")

        # LLM Note Generation
        st.subheader("🧠 Generate Clinical Note")

        prompt = f"""Patient Name: {patient_name}
Modality: {modality}
Scan Date: {scan_date}
Findings: Image uploaded from DICOM scan. Please generate a clinical radiology note."""

        if st.button("📋 Generate Note"):
            try:
                res = requests.post(
                    "http://localhost:1234/v1/chat/completions",
                    json={
                        "model": "openai/gpt-oss-20b",
                        "messages": [
                            {"role": "system", "content": "You are a radiology assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 512,
                        "stream": False
                    },
                    timeout=30
                )
                if res.status_code == 200:
                    note = res.json().get("choices", [{}])[0].get("message", {}).get("content", "No output")
                    st.text_area("📋 Clinical Note", note, height=300)
                else:
                    st.error(f"LLM failed with status {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"❌ LLM failed: {e}")

    except Exception as e:
        st.error(f"❌ Failed to process DICOM: {e}")
