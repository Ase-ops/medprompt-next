import streamlit as st
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from monai.transforms import LoadImage
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, Resize
import torch
from transformers import pipeline

st.set_page_config(page_title="MedPrompt: DICOM Analyzer", layout="wide")
st.title("🩻 MedPrompt: DICOM Analyzer + Note Generator")

# --- Upload DICOM File ---
dcm_file = st.file_uploader("📁 Choose a DICOM (.dcm) file", type=["dcm"])
if dcm_file:
    try:
        ds = pydicom.dcmread(dcm_file)
        st.success("✅ DICOM file read successfully.")

        # --- Metadata ---
        metadata = {
            "Patient Name": str(ds.get("PatientName", "")),
            "Modality": str(ds.get("Modality", "")),
            "Scan Date": str(ds.get("StudyDate", ""))
        }
        st.subheader("📄 DICOM Metadata")
        st.json(metadata)

        # --- Preview Image ---
        try:
            img = ds.pixel_array
            fig, ax = plt.subplots()
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            st.subheader("🖼️ Scan Preview")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"❗ Image preview failed: {e}")

        # --- MONAI Inference ---
        st.subheader("🧬 MONAI Inference (Dummy Segmentation)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert DICOM to Numpy
        image = img.astype(np.float32)
        image = np.expand_dims(image, axis=(0, 1))  # Shape: [1, 1, H, W]
        image = torch.tensor(image).to(device)

        # Dummy UNet model for segmentation
        model = UNet(
            dimensions=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
        ).to(device)
        model.eval()

        with torch.no_grad():
            output = model(image)
            mask = output[0][0].cpu().numpy()
            fig2, ax2 = plt.subplots()
            ax2.imshow(mask, cmap="jet", alpha=0.6)
            st.pyplot(fig2)

        # --- Clinical Note Generation ---
        st.subheader("🧠 Generate Clinical Note")
        generate_note = st.button("📝 Generate Note")
        if generate_note:
            # Use a local HuggingFace model (free + offline-capable)
            llm = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=0 if torch.cuda.is_available() else -1)

            prompt = f"""
            Patient modality: {metadata['Modality']}
            Scan date: {metadata['Scan Date']}
            Findings from scan: <dummy mask present>
            Write a short clinical radiology note for this scan:
            """
            response = llm(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]
            st.text_area("🧾 Clinical Note", value=response, height=150)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
