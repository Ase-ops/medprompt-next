import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import io
import base64

st.set_page_config(page_title="🩻 MedPrompt: DICOM Analyzer", layout="wide")

st.title("🩻 MedPrompt: DICOM File Analyzer")
st.markdown("Upload a `.dcm` file to extract metadata and visualize the scan.")

uploaded = st.file_uploader("Choose a DICOM (.dcm) file", type=["dcm"])

if uploaded:
    try:
        ds = pydicom.dcmread(uploaded)
        st.success("✅ DICOM file read successfully.")

        # Extract metadata
        name = str(getattr(ds, "PatientName", ""))
        modality = str(getattr(ds, "Modality", ""))
        date = str(getattr(ds, "StudyDate", getattr(ds, "AcquisitionDate", "")))

        st.subheader("📄 DICOM Metadata")
        st.json({
            "Patient Name": name,
            "Modality": modality,
            "Scan Date": date
        })

        # Image Preview
        if hasattr(ds, "pixel_array"):
            arr = ds.pixel_array
            norm = (arr - arr.min()) / (arr.max() - arr.min())
            uint8 = (norm * 255).astype(np.uint8)
            img = Image.fromarray(uint8)
            st.subheader("🖼️ Scan Preview")
            st.image(img)
        else:
            st.warning("No pixel data found.")

    except Exception as e:
        st.error(f"❌ Error processing DICOM: {e}")
