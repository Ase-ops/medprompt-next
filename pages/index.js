import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
  };

  const handleAnalyze = async () => {
    if (!file) return alert("⚠️ Upload a file first");
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("dicom_file", file);
      formData.append("prompt", "Analyze this scan for abnormalities.");

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      
      if (!res.ok || data.error) {
        alert(`Error: ${data.error || "Failed to analyze file"}`);
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 32 }}>
      <h1>🩺 MedPrompt UI</h1>
      <input type="file" accept=".dcm" onChange={handleUpload} />
      <button onClick={handleAnalyze} disabled={loading}>
        🔍 Analyze
      </button>
      {loading && <p>Analyzing…</p>}
      {result && (
        <div style={{ marginTop: 24 }}>
          <p><strong>Findings:</strong> {result.findings}</p>
          <p><strong>Confidence:</strong> {result.ai_confidence}</p>
          {result.image_preview_base64 && (
            <img
              src={`data:image/png;base64,${result.image_preview_base64}`}
              alt="Preview"
              style={{ marginTop: 16, maxWidth: "100%" }}
            />
          )}
        </div>
      )}
    </div>
  );
}


