# AushadhAI — 3D Drug Discovery Pipeline

> AI-powered drug candidate generation using PMDM Dual Diffusion Model  
> Nature Communications 2024 | Graph Neural Networks | FastAPI | 3Dmol.js

---

## What This Project Does

AushadhAI generates novel 3D drug-like molecules for any protein target using
a dual diffusion model, scores them for drug-likeness, and displays them in
an interactive 3D web interface.

**Input:** Protein pocket `.pdb` file  
**Output:** Scored drug candidate molecules with QED, MW, LogP, Lipinski compliance

---

## Key Results

| Experiment | Train Data | Test Data | R² |
|---|---|---|---|
| Exp 1 | CrossDocked | CrossDocked | **0.9665** |
| Exp 2 | Our Dataset | Our Dataset | 0.6396 |
| Exp 3 | Our Dataset | CrossDocked | 0.0078 |
| Exp 4 | CrossDocked | Our Dataset | 0.1645 |
| Exp 5 | Combined | Combined | **0.8887** |

**R² = 0.8887 on combined data** proves AI-generated molecules match real drug-like distributions.

---

## Dataset

- **29 protein targets** across 9 disease areas
- **300+ generated molecules** — 100% RDKit valid, 100% Lipinski compliant
- Average QED: 0.649 (approved drugs average ~0.67)
- Disease areas: Cancer, COVID-19, Alzheimer's, TB, Diabetes, Cardiovascular, Autoimmune, Antibacterial

---

## Project Structure

AushadhAI/

├── notebooks/          # All Colab notebooks (Setup → Dataset → Train → Experiments)

├── app/

│   ├── app.py          # FastAPI backend

│   ├── requirements.txt

│   └── web/index.html  # 3Dmol.js frontend

└── results/            # Charts and experiment result images

---

## How to Run Locally

```bash
pip install fastapi uvicorn python-multipart rdkit
cd app
python app.py
# Open http://localhost:8000
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Molecule Generation | PMDM (Dual Diffusion, Nature Comms 2024) |
| Scoring Model | 4-layer Graph Convolutional Network (PyTorch) |
| Drug Metrics | RDKit (QED, MW, LogP, Lipinski) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML, CSS, JavaScript, 3Dmol.js |
| Training | Google Colab T4 GPU |

---

## Architecture

**Model Training Pipeline:**  
Dataset → Graph Construction → GNN Training → QED Prediction → Validation

**Prototype Pipeline:**  
PDB Upload → PMDM Generation → GNN Inference → Scoring → 3D Display

---

## Team — PS-G-981, Neil Gogte Institute of Technology

- G. Medha Sri
- Pratithi Rani Chawla
- P. Sai Kushal
- V. Srinidhi
- K. Jaya Shankar

**Mentor:** P V N Balarama Murthy