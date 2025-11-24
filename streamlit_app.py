import streamlit as st
import pydicom
from PIL import Image
import numpy as np
import io

st.title("🩻 MedPrompt: DICOM File Analyzer")
st.write("Upload a .dcm file to extract metadata and visualize the scan.")

uploaded_file = st.file_uploader("Choose a DICOM (.dcm) file", type="dcm")

if uploaded_file:
    try:
        ds = pydicom.dcmread(uploaded_file)
        st.success("DICOM file read successfully.")
        
        # Show metadata
        st.subheader("📄 DICOM Metadata")
        st.json({
            "Patient Name": str(ds.get("PatientName", "N/A")),
            "Modality": str(ds.get("Modality", "N/A")),
            "Scan Date": str(ds.get("StudyDate", "N/A")),
        })

        # Convert pixel data to image
        if 'PixelData' in ds:
            arr = ds.pixel_array
            image = Image.fromarray(arr)
            st.subheader("🖼️ Scan Preview")
            st.image(image, caption="DICOM Image", use_column_width=True)
        else:
            st.warning("No image data found in this DICOM file.")

    except Exception as e:
        st.error(f"Error processing DICOM: {e}")
