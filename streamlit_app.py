import streamlit as st
import pydicom
import numpy as np
import PIL.Image
import torch
from monai.transforms import (
    Compose, LoadImage, AddChannel, Resize,
    ScaleIntensity, ToTensor
)
from monai.networks.nets import UNet
from transformers import pipeline

# ------------------ Load Open Source LLM -------------------
@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", trust_remote_code=True)

llm = load_llm()

# ------------------ MONAI Model Setup ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2
).to(device)
model.eval()

# ------------------ UI ---------------------
st.title("🩻 MedPrompt: DICOM Analyzer + Note Generator")
uploaded_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])

if uploaded_file:
    st.success("✅ DICOM file read successfully.")
    ds = pydicom.dcmread(uploaded_file)

    st.subheader("📄 DICOM Metadata")
    st.json({
        "Patient Name": str(ds.get("PatientName", "")),
        "Modality": str(ds.get("Modality", "")),
        "Scan Date": str(ds.get("StudyDate", ""))
    })

    if hasattr(ds, "pixel_array"):
        img = ds.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img_pil = PIL.Image.fromarray((img * 255).astype(np.uint8)).convert("L")
        st.subheader("🖼️ Scan Preview")
        st.image(img_pil, caption="DICOM Image Preview", use_column_width=True)

        # MONAI Inference
        st.subheader("🧬 MONAI Inference (Dummy Segmentation)")
        transform = Compose([
            Resize((128, 128)),
            AddChannel(),
            ScaleIntensity(),
            ToTensor()
        ])
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]

        st.image(pred * 255, caption="MONAI Prediction", use_column_width=True)

        # LLM Clinical Note Generation
        st.subheader("🧠 Generate Startup Clinical Note")
        prompt = f"""
        Patient DICOM Scan:
        - Modality: {ds.get("Modality", "")}
        - Scan Date: {ds.get("StudyDate", "")}
        - Patient Name: {ds.get("PatientName", "")}
        - Inferred pattern: {np.unique(pred).tolist()}
        
        Generate a concise clinical radiology report.
        """
        if st.button("Generate Note"):
            note = llm(prompt, max_length=300)[0]['generated_text']
            st.text_area("📝 Startup Clinical Note", value=note, height=200)
    else:
        st.warning("❌ No image data found in DICOM.")
