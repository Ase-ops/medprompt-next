import streamlit as st
import pydicom
import numpy as np
from PIL import Image
import torch
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, Resize, ScaleIntensity, EnsureChannelFirst, ToTensor
)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Page title
st.title("🩻 MedPrompt: DICOM Analyzer + Note Generator")

# File uploader
uploaded_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])

if uploaded_file:
    # Load DICOM
    dicom = pydicom.dcmread(uploaded_file)

    st.success("✅ DICOM file read successfully.")

    # Extract metadata
    metadata = {
        "Patient Name": str(getattr(dicom, "PatientName", "")),
        "Modality": str(getattr(dicom, "Modality", "")),
        "Scan Date": str(getattr(dicom, "StudyDate", ""))
    }

    st.subheader("📄 DICOM Metadata")
    st.json(metadata)

    # Display image preview
    try:
        image = dicom.pixel_array
        img_pil = Image.fromarray(image).convert("L").resize((256, 256))
        st.subheader("🖼️ Scan Preview")
        st.image(img_pil)
    except Exception as e:
        st.error(f"❌ Error processing DICOM image: {e}")

    # MONAI inference (Dummy model for segmentation)
    st.subheader("🧬 MONAI Inference (Simulated)")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

        transforms = Compose([
            EnsureChannelFirst(),
            Resize((256, 256)),
            ScaleIntensity(),
            ToTensor()
        ])

        input_tensor = transforms(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            pred_img = Image.fromarray((pred * 255).astype(np.uint8))
            st.image(pred_img, caption="🧬 Segmentation Output")
    except Exception as e:
        st.warning(f"⚠️ MONAI inference skipped: {e}")

    # LLM NOTE GENERATION
    st.subheader("🧠 Generate Clinical Note")
    prompt = f"""You are a radiology assistant. Based on the following DICOM metadata:
Patient Name: {metadata['Patient Name']}
Modality: {metadata['Modality']}
Scan Date: {metadata['Scan Date']}
Generate a clinical summary:"""

    if st.button("📝 Generate Note"):
        with st.spinner("Generating..."):
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            output_ids = model.generate(**inputs, max_new_tokens=150)
            note = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            st.text_area("📋 Clinical Note", value=note, height=200)
