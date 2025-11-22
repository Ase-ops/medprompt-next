// pages/api/analyze.js

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Only POST allowed" });

  const chunks = [];
  req.on("data", chunk => chunks.push(chunk));
  req.on("end", () => {
    const buffer = Buffer.concat(chunks);
    const base64 = buffer.toString("base64");

    res.status(200).json({
      findings: "🧠 Mock: No tumors detected.",
      ai_confidence: "92.4%",
      image_preview_base64: base64.slice(0, 300), // shorten for preview
    });
  });
}
