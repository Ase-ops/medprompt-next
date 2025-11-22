// pages/api/analyze.js

import { spawnSync } from 'child_process';
import formidable from 'formidable';

export const config = {
  api: {
    bodyParser: false,
  },
};

function parseForm(req) {
  return new Promise((resolve, reject) => {
    const form = new formidable.IncomingForm({ multiples: false });
    form.parse(req, (err, fields, files) => {
      if (err) return reject(err);
      resolve({ fields, files });
    });
  });
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { files } = await parseForm(req);
    const file = files.file || files.dicom || Object.values(files)[0];
    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // formidable v2 uses filepath, older versions use path
    const filePath = file.filepath || file.path || file.file;
    if (!filePath) {
      return res.status(400).json({ error: 'Uploaded file path not found' });
    }

    const py = `import sys, json, base64
from pydicom import dcmread
from PIL import Image
import io
fn = sys.argv[1]
try:
    ds = dcmread(fn)
except Exception as e:
    print(json.dumps({'__error__': 'dcmread_failed', 'message': str(e)}))
    sys.exit(0)

def _to_str(x):
    try:
        return str(x)
    except:
        return ''

patient = _to_str(getattr(ds, 'PatientName', ''))
modality = _to_str(getattr(ds, 'Modality', ''))
scan_date = _to_str(getattr(ds, 'StudyDate', getattr(ds, 'AcquisitionDate', '')))
result = {'patient': patient, 'modality': modality, 'scan_date': scan_date}

if hasattr(ds, 'PixelData'):
    try:
        arr = ds.pixel_array
        if arr is not None:
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            result['image_base64'] = b64
    except Exception as e:
        result['image_error'] = str(e)

print(json.dumps(result))`;

    const proc = spawnSync('python3', ['-c', py, filePath], { encoding: 'utf8', maxBuffer: 50 * 1024 * 1024 });

    if (proc.error) {
      return res.status(500).json({ error: 'Failed to run python subprocess', details: String(proc.error) });
    }

    const out = proc.stdout && proc.stdout.trim();
    const err = proc.stderr && proc.stderr.trim();

    if (!out) {
      return res.status(500).json({ error: 'No output from python', stderr: err || null });
    }

    let parsed;
    try {
      parsed = JSON.parse(out);
    } catch (e) {
      return res.status(500).json({ error: 'Invalid JSON from python', stdout: out, stderr: err });
    }

    // If python reported a dcmread error, return 400
    if (parsed && parsed.__error__ === 'dcmread_failed') {
      return res.status(400).json({ error: 'Failed to read DICOM', message: parsed.message });
    }

    const response = {
      patient: parsed.patient || null,
      modality: parsed.modality || null,
      scan_date: parsed.scan_date || null,
      image_base64: parsed.image_base64 || null,
    };

    return res.status(200).json(response);
  } catch (err) {
    console.error('analyze API error:', err);
    return res.status(500).json({ error: 'Internal server error', details: String(err) });
  }
}
