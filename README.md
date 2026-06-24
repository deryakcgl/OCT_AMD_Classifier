OCT AMD Classifier

A personal project I built to explore OCT image classification and a small clinical follow-up workflow.

The pipeline takes a retinal OCT scan, classifies it into CNV, DME, DRUSEN, or NORMAL, and when a patient has more than one visit uses Ollama locally to generate a short English comparison summary for review.

This is a learning and demo project, not a medical device. All outputs require physician review.

What it does

1. Train / export a small CNN on OCT images (PyTorch → ONNX)
2. Store patients and visits in SQLite
3. Run inference on new uploads through the desktop app
4. Compare the first and latest visit with a local LLM when results change (or on demand)

The LLM does not see raw images, only classifier outputs, dates, and complaint text.

What is not in this repository


data/, archive/  - Dataset is too large 
octnet_fp32.pth, octnet.onnx, octnet.onnx.data - Regenerate with training scripts
oct_clinic.db - Local demo database 

Run the clinic app

```bash
cd oct-amd-classifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull llama3  

cd src
python seed_data.py  # demo patients: Reyhan KO, Metin GU, NECLA DA
python app.py
```

You need `octnet.onnx` in the project root (from `export_onnx.py` after training).

Optional environment variables:

```bash
export OLLAMA_MODEL=llama3
export OLLAMA_BASE_URL=http://localhost:11434
```

On macOS with Homebrew Python, tkinter may require: `brew install python-tk@3.14`


Training workflow

If you have the OCT2017 data prepared:

```bash
pip install -r requirements-train.txt
```

Suggested order:


make_subset.py      - balanced subset into data/
data_check.py       - verify loaders and tensor shapes
train.py            - octnet_fp32.pth
measure_baseline.py - fp32 size, speed, accuracy
confusion.py        - confusion matrix plot
quantize.py         - int8 experiment (not used by the app)
export_onnx.py      - octnet.onnx
onnx_infer.py       - onnx accuracy check



Project structure

src/
  model.py            OCTNet architecture (shared across scripts)
  train.py            Training
  export_onnx.py      ONNX export
  inference.py        Single-image inference for the app
  db.py                 SQLite patients and visits
  patient_storage.py    Copies uploads into data/patients/
  genai_compare.py      Ollama prompts and visit comparison
  app.py                Tkinter desktop UI
  seed_data.py          Demo patient data
  test_compare.py       CLI test for GenAI without opening the UI




Design notes

- Compare uses the first visit vs the most recent one; intermediate visits are not included yet.
- Uploaded images are copied into `data/patients/` so paths remain valid if the original file is moved.
- Ollama runs off the UI thread so the window stays responsive during report generation.
- `octnet_int8.pth` from `quantize.py` is experimental; the app uses ONNX at runtime.

Possible next steps: global average pooling in the CNN, eye (left/right) field in the database, consecutive visit comparison.



Disclaimer

For educational purposes only. Not intended for diagnosis or treatment decisions.
