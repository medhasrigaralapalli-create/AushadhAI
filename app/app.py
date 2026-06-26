"""
backend/app.py  —  PMDM Drug Discovery Backend
================================================
Run this locally on your laptop:
    pip install fastapi uvicorn python-multipart rdkit
    python app.py

Then open http://localhost:8000 in your browser.

HOW IT WORKS:
- Frontend sends a PDB file + number of atoms
- Backend calls your PMDM model (via Colab OR uses pre-generated demo SDFs)
- Returns molecule data (SMILES, QED, MW, LogP) as JSON
- Frontend shows the results as 3D molecules + a table
"""

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pathlib, json, shutil, uuid, os, subprocess

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent
WEB_DIR     = BASE_DIR / "web"
UPLOADS_DIR = BASE_DIR / "uploads"
JOBS_DIR    = BASE_DIR / "jobs"
GEN_SOURCE  = BASE_DIR / "generated" / "source"   # pre-generated demo SDFs go here

for d in [UPLOADS_DIR, JOBS_DIR, GEN_SOURCE]:
    d.mkdir(parents=True, exist_ok=True)

# ── Google Drive / Colab paths (change these to match YOUR Drive) ──────────
DRIVE_BASE = pathlib.Path("/content/drive/MyDrive/PMDM")
COLAB_PYTHON = "/content/micromamba/envs/pmdm/bin/python"
COLAB_CKPT   = DRIVE_BASE / "checkpoints" / "500.pt"
COLAB_REPO   = DRIVE_BASE / "repo"

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="PMDM Drug Discovery API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────────────────────
def compute_metrics_rdkit(sdf_path: pathlib.Path) -> dict:
    """Compute drug-likeness metrics using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED
        mol = Chem.MolFromMolFile(str(sdf_path), sanitize=False)
        if mol is None:
            return {"valid": False}
        Chem.SanitizeMol(mol)
        mw   = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd  = Descriptors.NumHDonors(mol)
        hba  = Descriptors.NumHAcceptors(mol)
        qed  = QED.qed(mol)
        lip  = sum([mw <= 500, logp <= 5, hbd <= 5, hba <= 10])
        return {
            "valid":    True,
            "smiles":   Chem.MolToSmiles(mol),
            "qed":      round(qed, 3),
            "mw":       round(mw, 1),
            "logp":     round(logp, 2),
            "hbd":      hbd,
            "hba":      hba,
            "lipinski": lip,
            "num_atoms": mol.GetNumAtoms(),
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


def get_sdf_content(sdf_path: pathlib.Path) -> str:
    """Read SDF file content for 3D viewer."""
    try:
        return sdf_path.read_text(errors="ignore")
    except Exception:
        return ""


def build_molecule_result(sdf_path: pathlib.Path, index: int, job_id: str) -> dict:
    """Build a complete molecule result dict from an SDF file."""
    metrics = compute_metrics_rdkit(sdf_path)
    sdf_content = get_sdf_content(sdf_path)
    return {
        "id":          index,
        "name":        sdf_path.stem,
        "sdf_name":    sdf_path.name,
        "sdf_content": sdf_content,
        "sdf_url":     f"/sdf/{job_id}/{sdf_path.name}",
        **metrics,
    }


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Redirect to UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/")


@app.get("/health")
def health():
    """Check if backend is running."""
    # Check if RDKit is available
    try:
        from rdkit import Chem
        rdkit_ok = True
    except ImportError:
        rdkit_ok = False

    demo_sdfs = list(GEN_SOURCE.glob("*.sdf"))
    return {
        "status":      "running",
        "rdkit":       rdkit_ok,
        "demo_sdfs":   len(demo_sdfs),
        "mode":        "demo" if demo_sdfs else "no_demo_sdfs",
        "message":     "Ready" if demo_sdfs else "Add .sdf files to generated/source/ for demo mode",
    }


@app.post("/generate")
async def generate(
    pdb_file:    UploadFile = File(...),
    num_atom:    int = Form(20),
    num_samples: int = Form(3),
):
    """
    Main endpoint: accepts PDB file, returns generated molecules.

    DEMO MODE: If generated/source/ has .sdf files, serves those.
    LIVE MODE: Calls actual PMDM model (requires Colab to be running).
    """
    if not pdb_file.filename.lower().endswith(".pdb"):
        raise HTTPException(400, "Only .pdb files accepted")

    job_id = uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()

    # Save uploaded PDB
    pdb_path = job_dir / pdb_file.filename
    with open(pdb_path, "wb") as f:
        shutil.copyfileobj(pdb_file.file, f)

    results_data = []
    mode = "unknown"

    # ── DEMO MODE: serve pre-generated SDFs ──────────────────────────────
    # Cycles through available SDFs if num_samples exceeds files on disk
    demo_sdfs = sorted(GEN_SOURCE.glob("*.sdf"))
    if demo_sdfs:
        mode = "demo"
        selected = [demo_sdfs[i % len(demo_sdfs)] for i in range(num_samples)]
        seen_names = {}
        for i, sdf in enumerate(selected):
            stem = sdf.stem
            if stem in seen_names:
                seen_names[stem] += 1
                dest_name = f"{stem}_v{seen_names[stem]}{sdf.suffix}"
            else:
                seen_names[stem] = 0
                dest_name = sdf.name
            dest = job_dir / dest_name
            shutil.copy2(sdf, dest)
            result = build_molecule_result(dest, i, job_id)
            results_data.append(result)

    # ── LIVE MODE: call actual PMDM ──────────────────────────────────────
    elif COLAB_REPO.exists() and COLAB_CKPT.exists():
        mode = "live"
        try:
            out_dir = job_dir / "generate_ref"
            out_dir.mkdir()

            # Run PMDM
            r = subprocess.run(
                [
                    str(COLAB_PYTHON),
                    str(COLAB_REPO / "sample_for_pdb.py"),
                    "--ckpt",          str(COLAB_CKPT),
                    "--pdb_path",      str(pdb_path),
                    "--num_atom",      str(num_atom),
                    "--num_samples",   str(num_samples),
                    "--save_sdf",      "True",
                    "--sampling_type", "generalized",
                    "--batch_size",    "2",
                ],
                capture_output=True, text=True,
                cwd=str(COLAB_REPO), timeout=600
            )
            sdfs = sorted(out_dir.glob("*.sdf"))
            for i, sdf in enumerate(sdfs):
                result = build_molecule_result(sdf, i, job_id)
                results_data.append(result)
        except Exception as e:
            mode = "live_failed"
            results_data = []

    # Save job result
    job_record = {
        "job_id":      job_id,
        "status":      "done" if results_data else "failed",
        "mode":        mode,
        "pdb_file":    pdb_file.filename,
        "num_atom":    num_atom,
        "num_samples": num_samples,
        "count":       len(results_data),
        "molecules":   results_data,
    }
    (job_dir / "result.json").write_text(json.dumps(job_record, indent=2))

    if not results_data:
        raise HTTPException(500,
            "No molecules generated. Add .sdf files to generated/source/ for demo mode.")

    return job_record


@app.get("/job/{job_id}")
def get_job(job_id: str):
    """Get results of a previous job."""
    result_file = JOBS_DIR / job_id / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "Job not found")
    return json.loads(result_file.read_text())


@app.get("/sdf/{job_id}/{filename}")
def serve_sdf(job_id: str, filename: str):
    """Serve an SDF file for download."""
    sdf_path = JOBS_DIR / job_id / filename
    if not sdf_path.exists():
        raise HTTPException(404, "SDF file not found")
    return FileResponse(str(sdf_path), filename=filename)


# ── Serve frontend ─────────────────────────────────────────────────────────
if (WEB_DIR / "index.html").exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")
else:
    @app.get("/ui/")
    def no_ui():
        return {"error": "Frontend not found. Put index.html in the web/ folder."}


# ── Run directly ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("  PMDM Backend starting...")
    print("  Open http://localhost:8000 in your browser")
    print("="*50 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
