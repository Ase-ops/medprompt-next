import streamlit as st
import pydicom
import numpy as np
import torch
from PIL import Image
from monai.transforms import (
    Compose, LoadImage, AddChannel, Resize, ScaleIntensity, ToTensor
)
from monai.networks.nets import UNet
from transformers import pipeline

# --- Setup ---
st.set_page_config(page_title="MedPrompt: DICOM Analyzer", layout="wide")
st.title("🩻 MedPrompt: DICOM Analyzer + Note Generator")

# --- Upload DICOM ---
uploaded_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])

if uploaded_file:
    # Read DICOM file
    dicom = pydicom.dcmread(uploaded_file)
    st.success("✅ DICOM file read successfully.")
    
    # Display Metadata
    metadata = {
        "Patient Name": str(dicom.get("PatientName", "")),
        "Modality": str(dicom.get("Modality", "")),
        "Scan Date": str(dicom.get("StudyDate", ""))
    }
    st.subheader("📄 DICOM Metadata")
    st.json(metadata)

    # Preview Image
    try:
        image = dicom.pixel_array.astype(float)
        image_scaled = (np.maximum(image, 0) / image.max()) * 255.0
        image_scaled = np.uint8(image_scaled)
        image_pil = Image.fromarray(image_scaled).convert("L").resize((256, 256))
        st.subheader("🖼️ DICOM Image Preview")
        st.image(image_pil)
    except Exception as e:
        st.warning(f"⚠️ Image preview failed: {e}")

    # --- MONAI Inference ---
    st.subheader("🧬 MONAI Inference (UNet Segmentation)")
    try:
        transform = Compose([
            AddChannel(),
            Resize((256, 256)),
            ScaleIntensity(),
            ToTensor()
        ])
        image_tensor = transform(image_scaled).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        ).to(device)

        model.eval()
        with torch.no_grad():
            output = model(image_tensor.to(device))
        st.success("✅ MONAI segmentation inference complete.")
    except Exception as e:
        st.error(f"⚠️ MONAI Error: {e}")

    # --- Open-Source LLM Clinical Note Generator ---
    st.subheader("🧠 Generate Clinical Note (Open LLM)")
    if st.button("Generate Note"):
        with st.spinner("Generating..."):
            try:
                generator = pipeline("text-generation", model="openai-community/gpt2", device=0 if torch.cuda.is_available() else -1)
                prompt = f"CT scan of {metadata['Patient Name']} on {metadata['Scan Date']} shows..."
                note = generator(prompt, max_length=100, do_sample=True)[0]["generated_text"]
                st.text_area("📝 Clinical Note", note, height=200)
            except Exception as e:
                st.error(f"LLM error: {e}")
