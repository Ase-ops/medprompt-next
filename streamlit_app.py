import streamlit as st
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import numpy as np
import io
import json
import requests

st.set_page_config(page_title="🩻 MedPrompt", layout="wide")

st.title("🩻 MedPrompt: DICOM File Analyzer + Note Generator")
st.write("Upload a `.dcm` file to extract metadata, preview scan, and generate LLM-powered notes.")

uploaded_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])

if uploaded_file:
    try:
        dicom_data = pydicom.dcmread(uploaded_file)
        st.success("✅ DICOM file read successfully.")

        # Extract metadata
        patient_name = getattr(dicom_data, "PatientName", "")
        modality = getattr(dicom_data, "Modality", "")
        scan_date = getattr(dicom_data, "StudyDate", "")

        metadata = {
            "Patient Name": str(patient_name),
            "Modality": modality,
            "Scan Date": scan_date
        }

        st.subheader("📄 DICOM Metadata")
        st.json(metadata)

        # Convert to image if possible
        if "PixelData" in dicom_data:
            img_array = apply_voi_lut(dicom_data.pixel_array, dicom_data)
            if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                img_array = np.amax(img_array) - img_array

            img = Image.fromarray(img_array)
            img = img.convert("L")  # Convert to grayscale
            st.subheader("🖼️ Scan Preview")
            st.image(img, caption="DICOM Image Preview", use_column_width=True)

        # --- LLM startup-style note generation ---
        st.subheader("🧠 Generate Startup Clinical Note")

        prompt = f"""
        Generate a clinical summary based on the following metadata:
        - Patient Name: {patient_name}
        - Modality: {modality}
        - Scan Date: {scan_date}
        Use formal medical language suitable for a startup clinical intake note.
        """

        if st.button("📝 Generate Note"):
            with st.spinner("Generating note using local LLM..."):
                # LOCAL LLM OPTION — via LM Studio (http://localhost:1234)
                try:
                    llm_response = requests.post(
                        "http://localhost:1234/v1/completions",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({
                            "prompt": prompt,
                            "max_tokens": 300,
                            "temperature": 0.7,
                            "stop": ["\n\n"],
                            "model": "TheBloke/zephyr-7B-GGUF"  # customize if needed
                        })
                    )
                    result = llm_response.json()
                    note = result.get("choices", [{}])[0].get("text", "").strip()
                    st.code(note or "⚠️ No output from model.")
                except Exception as e:
                    st.error("⚠️ Failed to connect to local LLM. Is LM Studio running?")
    except Exception as e:
        st.error(f"Error processing DICOM: {e}")
