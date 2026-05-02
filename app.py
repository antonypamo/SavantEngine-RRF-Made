# ======================================================
# Savant RRF Φ12.0 — app.py (AGIRRFCore-aligned, HARDENED)
# Uses the same AGIRRFCore logic as RRFSavant_AGI_Core_Colab
# ======================================================

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os, json, math, time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import joblib


# ======================================================
# 0) Hardening limits
# ======================================================

MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "8000"))
MAX_ANSWER_CHARS = int(os.environ.get("MAX_ANSWER_CHARS", "12000"))
MAX_DOCS        = int(os.environ.get("MAX_DOCS", "50"))
MAX_DOC_CHARS   = int(os.environ.get("MAX_DOC_CHARS", "6000"))


# ======================================================
# 1) MANIFEST
# ======================================================

DEFAULT_MANIFEST = {
    "version": "Φ12.0",
    "project": "Savant RRF API & Meta-Logic Suite",
    "owner": "Antony Padilla Morales",
    "status": "fallback_default",
}

MANIFEST_PATH = Path(__file__).parent / "savant_rrf_api_manifest_phi12.json"

def load_manifest_file() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            print(f"[Manifest] Loading from {MANIFEST_PATH}", flush=True)
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Manifest] Invalid JSON: {e}", flush=True)
    print("[Manifest] Using DEFAULT_MANIFEST", flush=True)
    return DEFAULT_MANIFEST

manifest_data = load_manifest_file()
print("[Manifest] version:", manifest_data.get("version"), flush=True)


# ======================================================
# 2) Global config
# ======================================================

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # set in Spaces secrets
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

ENCODER_MODEL_ID    = "antonypamo/RRFSAVANTMADE"
META_LOGIT_REPO     = "antonypamo/RRFSavantMetaLogicV2"
META_LOGIT_FILENAME = "logreg_rrf_savant.joblib"

RRF_DATASET_REPO = "antonypamo/savant_rrf1_curated"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st_device = "cuda" if torch.cuda.is_available() else "cpu"


def _hf_download_safe(
    repo_id: str,
    filename: str,
    *,
    repo_type: Optional[str] = None,
    token: Optional[str] = None,
) -> Optional[str]:
    """
    Robust HF download:
    - returns local path or None
    - prints actionable errors (401/private/gated/missing)
    """
    try:
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            token=token or None,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            print(f"❌ [HF] 401 Unauthorized downloading {repo_id}/{filename}. "
                  f"Repo may be private/gated or HF_TOKEN missing/invalid.", flush=True)
        elif "RepositoryNotFoundError" in msg or "404" in msg:
            print(f"❌ [HF] Repo or file not found: {repo_id}/{filename}", flush=True)
        else:
            print(f"⚠️ [HF] Download failed: {repo_id}/{filename} | {e}", flush=True)
        return None


def hf_dataset_path(filename: str) -> Optional[str]:
    return _hf_download_safe(
        repo_id=RRF_DATASET_REPO,
        filename=filename,
        repo_type="dataset",
        token=HF_TOKEN if HF_TOKEN else None,
    )


# ======================================================
# 3) Optional artifacts (dataset assets)
# ======================================================

SAVANT_CNN_PATH  = hf_dataset_path("savant_cnn.pt")
RRF_NODES_PATH   = hf_dataset_path("rrf_nodes.pt")
RRF_TUTOR_JSONL  = hf_dataset_path("rrf_tutor_curated.jsonl")


# ======================================================
# 4) Savant CNN (optional)
# ======================================================

class SavantCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(512, 64)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


savant_cnn = None
if SAVANT_CNN_PATH:
    try:
        savant_cnn = SavantCNN()
        savant_cnn.load_state_dict(torch.load(SAVANT_CNN_PATH, map_location=device))
        savant_cnn.to(device).eval()
        print("✅ Savant CNN loaded", flush=True)
    except Exception as e:
        print(f"⚠️ CNN load failed: {e}", flush=True)

rrf_nodes = None
if RRF_NODES_PATH:
    try:
        rrf_nodes = torch.load(RRF_NODES_PATH, map_location=device)
        print("✅ RRF nodes loaded", flush=True)
    except Exception as e:
        print(f"⚠️ RRF nodes load failed: {e}", flush=True)


# ======================================================
# 5) Φ-node ontology (8 nodes -> one-hot 8)
# ======================================================

@dataclass
class PhiNode:
    name: str
    description: str
    tags: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # runtime only

PHI_NODES: List[PhiNode] = [
    PhiNode("Φ0_seed",      "Genesis seed, core identity and origin.", ["genesis","identity","anchor"]),
    PhiNode("Φ1_relation",  "Relational bonding, dialogue, social meaning.", ["relation","dialogue"]),
    PhiNode("Φ2_resonance", "Signal resonance, harmonic alignment, coherence lift.", ["resonance","harmonics"]),
    PhiNode("Φ3_memory",    "Memory consolidation, retrieval, indexing.", ["memory","retrieval"]),
    PhiNode("Φ4_logic",     "Logical rigor, constraints, verification.", ["logic","verification"]),
    PhiNode("Φ5_creative",  "Creative synthesis, metaphor, generative jumps.", ["creative","synthesis"]),
    PhiNode("Φ6_alignment", "Ethical alignment and safety constraints.", ["alignment","ethics"]),
    PhiNode("Φ7_meta_agi",  "Meta-orchestrator that evaluates and routes flows.", ["meta","orchestration"]),
]
PHI_NAME_TO_IDX = {n.name: i for i, n in enumerate(PHI_NODES)}


def phi_nodes_public() -> List[Dict[str, Any]]:
    # JSON-safe version (no embeddings)
    return [{"name": n.name, "description": n.description, "tags": n.tags} for n in PHI_NODES]


# ======================================================
# 6) CoherenceModel (stable S_RRF + C_RRF)
# ======================================================

class CoherenceModel:
    def __init__(self, eps: float = 1e-9):
        self.eps = eps

    def compute(self, vec: np.ndarray) -> Tuple[float, float]:
        v = np.asarray(vec, dtype=float).ravel()
        n = len(v)
        if n < 4:
            return 0.0, 0.0

        spectrum = np.fft.rfft(v)
        power = (np.abs(spectrum) ** 2).astype(float)
        freqs = np.fft.rfftfreq(n, d=1.0).astype(float)

        total_power = float(power.sum()) + self.eps

        # C_RRF: concentration in dominant frequency
        C_RRF = float(power.max() / total_power)

        # S_RRF: prefer lower average frequency
        f_mean = float((freqs * power).sum() / total_power)
        f_max = float(freqs.max()) + self.eps
        S_RRF = float(1.0 - min(1.0, f_mean / f_max))

        return S_RRF, C_RRF

coherence_model = CoherenceModel()


# ======================================================
# 7) AGIRRFCore (aligned)
# ======================================================

class AGIRRFCore:
    def __init__(
        self,
        phi_nodes: List[PhiNode],
        coherence_model: Optional[CoherenceModel] = None,
        st_model_name: str = ENCODER_MODEL_ID,
    ):
        self.phi_nodes = phi_nodes
        self.coherence_model = coherence_model

        print(f"🔄 Loading sentence-transformer: {st_model_name} on {st_device} ...", flush=True)
        self.embedder = SentenceTransformer(st_model_name, device=st_device)
        print("✅ Embedder loaded", flush=True)

        self._embed_phi_nodes()

    def _embed_text(self, text: str) -> np.ndarray:
        return self.embedder.encode([text], convert_to_numpy=True)[0]

    def _embed_phi_nodes(self):
        texts = [f"{n.name}: {n.description} | tags: {', '.join(n.tags)}" for n in self.phi_nodes]
        embs = self.embedder.encode(texts, convert_to_numpy=True)
        for node, emb in zip(self.phi_nodes, embs):
            node.embedding = emb
        print(f"✅ Embedded {len(self.phi_nodes)} Φ-nodes.", flush=True)

    def _dominant_frequency(self, vec: np.ndarray) -> float:
        v = np.asarray(vec, dtype=float).ravel()
        if len(v) < 4:
            return 0.0
        spectrum = np.fft.rfft(v)
        power = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(len(v), d=1.0)
        idx = int(np.argmax(power))
        return float(freqs[idx])

    def _phi_omega(self, energy: float, dom_freq: float) -> Tuple[float, float]:
        phi = 1.0 - math.exp(-float(energy))      # saturating
        omega = math.tanh(dom_freq * 10.0)        # saturating
        return float(phi), float(omega)

    def _closest_phi_node(self, vec: np.ndarray) -> Tuple[str, float]:
        if not self.phi_nodes or self.phi_nodes[0].embedding is None:
            return "unknown", 0.0
        v = np.asarray(vec, dtype=float).ravel()
        v_norm = np.linalg.norm(v) + 1e-9
        best_name, best_cos = "unknown", -1.0
        for node in self.phi_nodes:
            e = node.embedding
            if e is None:
                continue
            cos = float(np.dot(v, e) / (v_norm * (np.linalg.norm(e) + 1e-9)))
            if cos > best_cos:
                best_cos = cos
                best_name = node.name
        return best_name, best_cos

    def analyze(self, text: str, context_label: str = "query") -> Dict[str, Any]:
        vec = self._embed_text(text)

        energy = float(np.dot(vec, vec))
        dom_freq = self._dominant_frequency(vec)
        phi, omega = self._phi_omega(energy, dom_freq)

        if self.coherence_model is not None:
            S_RRF, C_RRF = self.coherence_model.compute(vec)
        else:
            S_RRF, C_RRF = 0.0, 0.0

        coherence = 0.5 * float(S_RRF) + 0.5 * float(C_RRF)
        closest_name, closest_cos = self._closest_phi_node(vec)

        return {
            "context": context_label,
            "phi": phi,
            "omega": omega,
            "coherence": float(coherence),
            "S_RRF": float(S_RRF),
            "C_RRF": float(C_RRF),
            "hamiltonian_energy": float(energy),
            "dominant_frequency": float(dom_freq),
            "closest_phi_node": closest_name,
            "closest_phi_cos": float(closest_cos),
            "timestamp": float(time.time()),
        }


agirrf_core = AGIRRFCore(
    phi_nodes=PHI_NODES,
    coherence_model=coherence_model,
    st_model_name=ENCODER_MODEL_ID,
)


# ======================================================
# 8) Load Meta-Logit (15D)
# ======================================================

print("🔄 Loading meta-logit...", flush=True)
meta_logit_path = _hf_download_safe(
    repo_id=META_LOGIT_REPO,
    filename=META_LOGIT_FILENAME,
    token=HF_TOKEN if HF_TOKEN else None,
)
if not meta_logit_path:
    raise RuntimeError(
        f"Meta-logit not available. Check repo_id={META_LOGIT_REPO}, "
        f"filename={META_LOGIT_FILENAME}, and HF_TOKEN if private."
    )
meta_logit = joblib.load(meta_logit_path)

EXPECTED_FEATURES = getattr(meta_logit, "n_features_in_", 15)
if EXPECTED_FEATURES != 15:
    raise RuntimeError(f"Meta-logit expects {EXPECTED_FEATURES} features, expected 15.")
print("✅ Meta-logit ready (15D)", flush=True)


# ======================================================
# 9) Feature mapping (7 + one-hot 8 = 15)
# ======================================================

def rrf_state_to_features(state: Dict[str, Any]) -> np.ndarray:
    phi   = float(state.get("phi", 0.0))
    omega = float(state.get("omega", 0.0))
    coh   = float(state.get("coherence", 0.0))
    S_RRF = float(state.get("S_RRF", 0.0))
    C_RRF = float(state.get("C_RRF", 0.0))
    E_H   = float(state.get("hamiltonian_energy", 0.0))
    dom_f = float(state.get("dominant_frequency", 0.0))

    phi_name = state.get("closest_phi_node", "unknown")
    phi_onehot = np.zeros(len(PHI_NODES), dtype=float)
    idx = PHI_NAME_TO_IDX.get(phi_name)
    if idx is not None:
        phi_onehot[idx] = 1.0

    base = np.array([phi, omega, coh, S_RRF, C_RRF, E_H, dom_f], dtype=float)
    return np.concatenate([base, phi_onehot], axis=0)


# ======================================================
# 10) Core scoring (prompt, answer)
# ======================================================

def _embed_norm(text: str) -> np.ndarray:
    return agirrf_core.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]

def compute_scores(prompt: str, answer: str) -> Dict[str, Any]:
    prompt = prompt or ""
    answer = answer or ""
    if not prompt.strip() or not answer.strip():
        raise ValueError("Empty prompt/answer")

    if len(prompt) > MAX_PROMPT_CHARS or len(answer) > MAX_ANSWER_CHARS:
        raise HTTPException(status_code=413, detail="Payload too large")

    # extra signal: cosine(prompt, answer)
    e_p = _embed_norm(prompt)
    e_a = _embed_norm(answer)
    cosine = float(np.dot(e_p, e_a))

    # stable single-state features on combined QA text
    qa_text = f"Q: {prompt}\nA: {answer}"
    state = agirrf_core.analyze(qa_text, context_label="qa")
    feats = rrf_state_to_features(state).reshape(1, -1)

    p_good = float(meta_logit.predict_proba(feats)[0][1])

    SRRF = p_good
    CRRF = p_good * cosine
    E_phi = 0.5 * (p_good + abs(cosine))

    return {
        "p_good": p_good,
        "SRRF": SRRF,
        "CRRF": CRRF,
        "E_phi": E_phi,
        "cosine": cosine,

        # debug/state exposure (key for Savant)
        "phi": float(state["phi"]),
        "omega": float(state["omega"]),
        "coherence": float(state["coherence"]),
        "S_RRF": float(state["S_RRF"]),
        "C_RRF": float(state["C_RRF"]),
        "hamiltonian_energy": float(state["hamiltonian_energy"]),
        "dominant_frequency": float(state["dominant_frequency"]),
        "closest_phi_node": state["closest_phi_node"],
        "closest_phi_cos": float(state["closest_phi_cos"]),
    }


# ======================================================
# 11) FastAPI models
# ======================================================

class EvaluateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt: str
    answer: str
    model_label: Optional[str] = None  # reserved for future routing

class EvaluateResponse(BaseModel):
    scores: Dict[str, Any]
    manifest_version: str

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=15, max_length=15)

class PredictResponse(BaseModel):
    p_good: float

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    alpha: float = 0.2  # kept for compatibility (not used in cosine rerank)

class RerankDocument(BaseModel):
    id: int
    score: float
    rank: int

class RerankResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    results: List[RerankDocument]


# ======================================================
# 12) FastAPI app
# ======================================================

app = FastAPI(
    title="Savant RRF Φ12.0 API",
    version="1.2.1",
    description="AGIRRFCore-aligned Meta-Logic, Reranking & Quality Evaluation",
)


# --------------------------
# Root (avoid 404 in Spaces)
# --------------------------

@app.get("/")
def root():
    return {
        "status": "ok",
        "project": manifest_data.get("project"),
        "version": manifest_data.get("version"),
        "model": "RRFSavantMetaLogicV2",
        "docs": "/docs",
        "endpoints": ["/manifest", "/health", "/evaluate", "/predict", "/v1/rerank"],
    }


# --------------------------
# Manifest (no naming clash)
# --------------------------

@app.get("/manifest")
def get_manifest():
    return {
        "model": "RRFSavantMetaLogicV2",
        "version": manifest_data.get("version"),
        "encoder": ENCODER_MODEL_ID,
        "meta_logit": f"{META_LOGIT_REPO}/{META_LOGIT_FILENAME}",
        "features": 15,
        "phi_nodes": phi_nodes_public(),
        "limits": {
            "MAX_PROMPT_CHARS": MAX_PROMPT_CHARS,
            "MAX_ANSWER_CHARS": MAX_ANSWER_CHARS,
            "MAX_DOCS": MAX_DOCS,
            "MAX_DOC_CHARS": MAX_DOC_CHARS,
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "encoder_loaded": True,
        "meta_logit_loaded": True,
        "cnn_loaded": savant_cnn is not None,
        "rrf_nodes_loaded": rrf_nodes is not None,
        "manifest_version": manifest_data.get("version"),
        "phi_nodes": len(PHI_NODES),
        "device": str(device),
    }


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    try:
        scores = compute_scores(req.prompt, req.answer)
        return EvaluateResponse(scores=scores, manifest_version=str(manifest_data.get("version")))
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Evaluate] Error: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Evaluation failed")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        x = np.array([req.features], dtype=float)
        p_good = float(meta_logit.predict_proba(x)[0][1])
        return PredictResponse(p_good=p_good)
    except Exception as e:
        print(f"[Predict] Error: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Predict failed")


@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    try:
        if not req.query or not req.query.strip():
            raise HTTPException(status_code=400, detail="query is empty")

        if len(req.documents) > MAX_DOCS:
            raise HTTPException(status_code=413, detail="Too many documents")

        for d in req.documents:
            if len(d) > MAX_DOC_CHARS:
                raise HTTPException(status_code=413, detail="Document too large")

        texts = [req.query] + req.documents
        embs = agirrf_core.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        q_emb = embs[0]
        d_embs = embs[1:]
        scores = (d_embs @ q_emb).astype(float).tolist()

        results = [{"id": i, "score": float(s)} for i, s in enumerate(scores)]
        results.sort(key=lambda x: x["score"], reverse=True)

        ranked = [RerankDocument(id=r["id"], score=r["score"], rank=i + 1) for i, r in enumerate(results)]
        return RerankResponse(model_id=ENCODER_MODEL_ID, results=ranked)

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Rerank] Error: {e}", flush=True)
        raise HTTPException(status_code=500, detail="Rerank failed")
