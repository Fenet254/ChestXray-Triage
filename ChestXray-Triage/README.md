# ChestXray‑Triage — Deep Learning Capstone

## 🚀 Overview

This project is a **full‑stack deep learning system** for multi‑label chest X‑ray triage. It predicts multiple thoracic diseases (e.g., Pneumonia, Tuberculosis, Pleural Effusion) from chest radiographs, ranks urgency, and provides visual explanations with **Grad‑CAM heatmaps**. The project follows a **1‑month end‑to‑end workflow**: idea → dataset → model → improvements → evaluation → deployment → documentation → presentation.

⚠️ **Disclaimer**: This system is for **educational and research purposes only**. It is **not a medical diagnostic tool** and must not be used for clinical decision‑making.

---

## 📌 Goals

* Build a reproducible **computer vision pipeline** for chest X‑ray triage.
* Implement a **baseline model** and at least **three meaningful improvements**.
* Evaluate with AUROC, F1, PR‑AUC, confusion matrices, calibration, and error analysis.
* Deploy a working demo (web app with Streamlit/FastAPI).
* Deliver full documentation: Data Card, research paper, PPT deck, and recorded videos.

---


## 📂 Repository Structure

```
chestxray-triage/
  README.md
  RUN.md
  LICENSE
  data/
    raw/            # raw datasets (gitignored)
    processed/      # processed splits & metadata
    metadata.csv
    label_map.json
  configs/          # experiment configs (baseline, improvements)
  src/
    data/           # dataset & transform loaders
    models/         # model definitions
    app/            # deployment (Streamlit/FastAPI)
    train.py        # training script
    eval.py         # evaluation script
    gradcam.py      # Grad-CAM explainability
    export.py       # ONNX/TFLite export
    infer.py        # CLI inference
  scripts/          # preprocessing & setup scripts
  notebooks/        # EDA, error analysis, ablations
  tests/            # unit tests for reproducibility
  environment.yml   # conda environment file
  requirements.txt  # pip dependencies
  Makefile          # one-line workflows
```

---

## ⚙️ Quickstart

### 1. Setup environment

```bash
conda env create -f environment.yml
conda activate cxr
```

### 2. Prepare data

```bash
python scripts/preprocess.py --config configs/baseline.yaml
```

### 3. Train a baseline

```bash
python -m src.train --config configs/baseline.yaml
```

### 4. Evaluate

```bash
python -m src.eval --ckpt runs/baseline/best.ckpt --split test
```

### 5. Run Grad‑CAM

```bash
python -m src.gradcam --ckpt runs/final/best.ckpt --images demo/
```

### 6. Launch demo app

```bash
streamlit run src/app/streamlit_app.py
```

---

## 🧪 Model Improvements (Planned)

1. **Stronger backbone & higher resolution** (EfficientNetV2 / ConvNeXt).
2. **Loss functions for class imbalance** (Focal loss, class‑weighted BCE).
3. **Calibration & TTA** (temperature scaling, test‑time augmentations).

Optional: MixUp/CutMix, self‑supervised pretraining, lightweight ensembling.

---

## 📊 Evaluation Metrics

* **Macro AUROC** (primary)
* **Macro & per‑class F1**
* **Precision‑Recall AUC** for critical classes (TB, Pneumonia, Effusion)
* **Confusion matrices** & **error boards**
* **Calibration metrics** (ECE, reliability diagrams)
* **Latency measurements** (GPU vs CPU)

---

## 🖥️ Deployment

* **Streamlit UI**: Upload image → model predicts → urgency ranking + Grad‑CAM heatmaps.
* **FastAPI endpoint**: `/predict` returns JSON with per‑class probabilities, urgency score, latency.
* **ONNX/TFLite export**: Optional edge/mobile deployment with quantization.

---

## 📑 Deliverables

* ✅ Dataset (≥500 images, ≥40% curated/self‑collected)
* ✅ Baseline + 3 improvements
* ✅ Full evaluation & ablations
* ✅ Working demo (web app)
* ✅ Data Card (2 pages)
* ✅ Research Paper (PDF, template format)
* ✅ PPT deck (10–12 slides)
* ✅ Recorded videos (5 stages)

---

## 📅 Timeline

* **Week 1**: Define problem, collect pilot data, baseline.
* **Week 2**: Expand dataset, train baseline to convergence, start deployment.
* **Week 3**: Add improvements (backbone, loss, calibration), error analysis, export model.
* **Week 4**: Final evaluation, deployment polish, documentation, videos, submission.

---

## 📜 License & Citation

* Open‑source license (MIT or Apache 2.0 recommended).
* Cite all datasets (CheXpert, NIH, PadChest, etc.) and external libraries used.

---

## 🙏 Acknowledgements

* Public datasets: CheXpert, NIH ChestX‑ray14, PadChest, Shenzhen & Montgomery TB datasets.
* Libraries: PyTorch, Torchvision, Timm, Albumentations, Grad‑CAM, Streamlit, FastAPI.
