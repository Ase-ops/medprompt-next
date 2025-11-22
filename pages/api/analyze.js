import { IncomingForm } from "formidable";
import { spawnSync } from "child_process";
import fs from "fs";

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const form = new IncomingForm();

  return new Promise((resolve) => {
    form.parse(req, (err, fields, files) => {
      if (err) {
        res.status(500).json({ error: "Failed to parse form data" });
        return resolve();
      }

      const dicomFile = files.dicom_file;
      if (!dicomFile) {
        res.status(400).json({ error: "No DICOM file uploaded" });
        return resolve();
      }

      // Get the file path (handle both array and single file cases)
      const filePath = Array.isArray(dicomFile) ? dicomFile[0].filepath : dicomFile.filepath;

    // Python script to extract DICOM metadata and generate PNG preview
    const pythonScript = `
import sys
import json
import base64
from io import BytesIO

try:
    import pydicom
    from PIL import Image
except ImportError as e:
    print(json.dumps({"error": f"Missing dependency: {e}"}))
    sys.exit(1)

try:
    # Read DICOM file
    dcm = pydicom.dcmread(sys.argv[1])
    
    # Extract basic metadata
    metadata = {
        "PatientName": str(getattr(dcm, "PatientName", "N/A")),
        "PatientID": str(getattr(dcm, "PatientID", "N/A")),
        "StudyDate": str(getattr(dcm, "StudyDate", "N/A")),
        "Modality": str(getattr(dcm, "Modality", "N/A")),
        "StudyDescription": str(getattr(dcm, "StudyDescription", "N/A")),
    }
    
    # Generate PNG preview if pixel data exists
    preview_base64 = None
    if hasattr(dcm, "pixel_array"):
        pixel_array = dcm.pixel_array
        
        # Normalize to 0-255 range
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype("uint8")
        
        # Convert to PIL Image
        image = Image.fromarray(pixel_array)
        
        # Convert to PNG and encode as base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        preview_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    result = {
        "metadata": metadata,
        "image_preview_base64": preview_base64,
    }
    
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
`;

      // Execute Python script
      let pythonResult;
      try {
        pythonResult = spawnSync("python3", ["-c", pythonScript, filePath], {
          encoding: "utf-8",
          timeout: 10000,
        });
      } finally {
        // Clean up uploaded file
        try {
          fs.unlinkSync(filePath);
        } catch (cleanupErr) {
          console.error("Failed to clean up temp file:", cleanupErr);
        }
      }

      if (pythonResult.error) {
        res.status(500).json({
          error: "Failed to execute Python script",
          details: pythonResult.error.message,
        });
        return resolve();
      }

      if (pythonResult.status !== 0) {
        res.status(500).json({
          error: "Python script failed",
          stderr: pythonResult.stderr,
          stdout: pythonResult.stdout,
        });
        return resolve();
      }

      try {
        const result = JSON.parse(pythonResult.stdout);
        
        if (result.error) {
          res.status(500).json({
            error: "DICOM processing error",
            details: result.error,
          });
          return resolve();
        }

        // Return response in the format expected by the frontend
        res.status(200).json({
          findings: `📊 DICOM Metadata:\n` +
            `Patient: ${result.metadata.PatientName} (ID: ${result.metadata.PatientID})\n` +
            `Study Date: ${result.metadata.StudyDate}\n` +
            `Modality: ${result.metadata.Modality}\n` +
            `Description: ${result.metadata.StudyDescription}`,
          ai_confidence: "N/A",
          image_preview_base64: result.image_preview_base64,
        });
        return resolve();
      } catch (parseErr) {
        res.status(500).json({
          error: "Failed to parse Python output",
          details: parseErr.message,
          stdout: pythonResult.stdout,
        });
        return resolve();
      }
    });
  });
}
