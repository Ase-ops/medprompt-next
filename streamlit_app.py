import streamlit as st
import pydicom
import numpy as np
from PIL import Image
import io
import torch
from monai.transforms import Compose, LoadImage, AddChannel, Resize, ScaleIntensity, ToTensor
from monai.networks.nets import UNet

st.title("🩻 MedPrompt: DICOM Analyzer + Note Generator")

uploaded_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])

if uploaded_file:
    ds = pydicom.dcmread(uploaded_file)
    st.success("✅ DICOM file read successfully.")

    # 📄 Metadata
    st.subheader("📄 DICOM Metadata")
    metadata = {
        "Patient Name": str(ds.get("PatientName", "")),
        "Modality": str(ds.get("Modality", "")),
        "Scan Date": str(ds.get("StudyDate", "")),
    }
    st.json(metadata)

    # 🖼️ Preview
    st.subheader("🖼️ DICOM Image Preview")
    try:
        image_data = ds.pixel_array
        image_data = ((image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 255).astype(np.uint8)
        image = Image.fromarray(image_data)
        image = image.convert("L")
        st.image(image, caption="DICOM Image", use_column_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not render image: {e}")

    # 🧬 MONAI Inference
    st.subheader("🧬 MONAI Inference (Dummy Segmentation)")
    try:
        transform = Compose([
            AddChannel(),
            Resize((128, 128)),
            ScaleIntensity(),
            ToTensor()
        ])

        transformed = transform(image_data)
        transformed = transformed.unsqueeze(0)  # add batch dim

        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        )
        with torch.no_grad():
            model.eval()
            output = model(transformed)
        st.success("✅ MONAI dummy model ran successfully.")
        st.write("🧪 Output Shape:", output.shape)
    except Exception as e:
        st.error(f"⚠️ MONAI Error: {e}")

    # 🧠 LLM Note Generator (Stub)
    st.subheader("🧠 Generate Startup Clinical Note")
    if st.button("🧠 Generate Note"):
        st.markdown("🚧 *Note generator placeholder — connect open LLM backend here*")
        st.code("Findings: No acute abnormalities detected. Recommend follow-up in 6 months.")
