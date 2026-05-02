import os, sys, math
from typing import Optional, Dict, Any, List

import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import joblib

# ============================
# CONFIG
# ============================

HF_TOKEN = os.environ.get("HF_TOKEN", "")

ENCODER_MODEL_ID    = "antonypamo/RRFSAVANTMADE"
META_LOGIT_REPO     = "antonypamo/RRFSavantMetaLogicV2"
META_LOGIT_FILENAME = "logreg_rrf_savant.joblib"

MAX_PROMPT_CHARS = 8000
MAX_ANSWER_CHARS = 12000
MAX_DOCS = 50
MAX_DOC_CHARS = 6000

PHI_NODES = [
    "Φ0_seed",
    "Φ1_geometric",
    "Φ2_gauge_dirac",
    "Φ3_log_gravity",
    "Φ4_resonance",
    "Φ5_memory_symbiosis",
    "Φ6_alignment",
    "Φ7_meta_agi",
]

# ============================
# STARTUP: MODELS
# ============================

print("🔄 Loading encoder...", flush=True)
encoder = SentenceTransformer(ENCODER_MODEL_ID)
print("✅ Encoder loaded", flush=True)

print("🔄 Loading meta-logit V2...", flush=True)
meta_logit_path = hf_hub_download(
    repo_id=META_LOGIT_REPO,
    filename=META_LOGIT_FILENAME,
    token=HF_TOKEN or None,
)
meta_logit = joblib.load(meta_logit_path)

EXPECTED_FEATURES = getattr(meta_logit, "n_features_in_", 15)
if EXPECTED_FEATURES != 15:
    raise RuntimeError(f"Meta-logit expects {EXPECTED_FEATURES} features, expected 15.")

print("✅ Meta-logit loaded (15D)", flush=True)

# ============================
# META-STATE FEATURE EXTRACTION
# ============================

def get_embedding(text: str) -> np.ndarray:
    return encoder.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]


def spectral_features(emb: np.ndarray) -> Dict[str, float]:
    fft = np.fft.rfft(emb)
    power = np.abs(fft) ** 2

    total = power.sum() + 1e-12
    dominant_idx = int(np.argmax(power))

    phi = float(np.clip(total / (total + 1.0), 0.0, 1.0))
    omega = float(np.clip(dominant_idx / len(power), 0.0, 1.0))

    S_RRF = float(1.0 - np.std(power) / (np.mean(power) + 1e-12))
    S_RRF = float(np.clip(S_RRF, 0.0, 1.0))

    coherence = float(0.5 * (1.0 - np.std(power) / (np.mean(power) + 1e-12)) + 0.5 * C_RRF)

    hamiltonian_energy = float(np.dot(emb, emb))
    dominant_frequency = float(dominant_idx)

    return {
        "phi": phi,
        "omega": omega,
        "coherence": coherence,
        "S_RRF": S_RRF,
        "C_RRF": C_RRF,
        "hamiltonian_energy": hamiltonian_energy,
        "dominant_frequency": dominant_frequency,
    }


def closest_phi_node(feats: Dict[str, float]) -> int:
    # Deterministic ontology mapping
    if feats["coherence"] > 0.85 and feats["phi"] > 0.6:
        return 4  # Φ4_resonance
    if feats["hamiltonian_energy"] > 50:
        return 2  # Φ2_gauge_dirac
    if feats["omega"] < 0.2:
        return 0  # Φ0_seed
    if feats["coherence"] < 0.4:
        return 5  # Φ5_memory_symbiosis
    if feats["phi"] < 0.3:
        return 6  # Φ6_alignment
    return 7  # Φ7_meta_agi


def rrf_state_to_vector(prompt: str, answer: str) -> np.ndarray:
    emb = get_embedding(prompt + "\n" + answer)
    feats = spectral_features(emb)

    phi_idx = closest_phi_node(feats)
    phi_one_hot = [1.0 if i == phi_idx else 0.0 for i in range(8)]

    vector = [
        feats["phi"],
        feats["omega"],
        feats["coherence"],
        feats["S_RRF"],
        feats["C_RRF"],
        feats["hamiltonian_energy"],
        feats["dominant_frequency"],
        *phi_one_hot,
    ]

    return np.array(vector, dtype=float), feats, PHI_NODES[phi_idx]

# ============================
# FASTAPI
# ============================

app = FastAPI(
    title="Savant RRF Φ12.0 API",
    version="2.0.0",
    description="Meta-state RRF quality evaluation + rerank",
)

# ============================
# SCHEMAS
# ============================

class EvaluateRequest(BaseModel):
    prompt: str
    answer: str


class EvaluateResponse(BaseModel):
    p_good: float
    scores: Dict[str, float]
    features: Dict[str, float]
    phi_node: str


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    alpha: float = 0.2


class RerankDocument(BaseModel):
    id: int
    score: float
    rank: int


class RerankResponse(BaseModel):
    model_id: str
    results: List[RerankDocument]

# ============================
# MANIFEST / HEALTH
# ============================
@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "RRFSavantMetaLogicV2",
        "version": "Φ12.0",
        "docs": "/docs",
        "endpoints": ["/manifest", "/health", "/evaluate", "/quality", "/v1/rerank"],
    }

@app.get("/manifest")
def manifest():
    return {
        "model": "RRFSavantMetaLogicV2",
        "version": "Φ12.0",
        "encoder": ENCODER_MODEL_ID,
        "features": 15,
        "phi_nodes": PHI_NODES,
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# ============================
# /EVALUATE
# ============================

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    if len(req.prompt) > MAX_PROMPT_CHARS or len(req.answer) > MAX_ANSWER_CHARS:
        raise HTTPException(413, "Payload too large")

    x, feats, phi_node = rrf_state_to_vector(req.prompt, req.answer)
    proba = meta_logit.predict_proba(x.reshape(1, -1))[0]
    p_good = float(proba[1])

    scores = {
        "SRRF": p_good,
        "CRRF": p_good * feats["coherence"],
        "E_phi": 0.5 * (p_good + feats["phi"]),
    }

    return EvaluateResponse(
        p_good=p_good,
        scores=scores,
        features=feats,
        phi_node=phi_node,
    )


@app.post("/quality", response_model=EvaluateResponse)
def quality_alias(req: EvaluateRequest):
    return evaluate(req)

# ============================
# /v1/rerank (BATCHED)
# ============================

@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    if len(req.documents) > MAX_DOCS:
        raise HTTPException(413, "Too many documents")

    texts = [req.query] + req.documents
    for d in req.documents:
        if len(d) > MAX_DOC_CHARS:
            raise HTTPException(413, "Document too large")

    embs = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    q_emb = embs[0]
    d_embs = embs[1:]

    scores = d_embs @ q_emb
    ranked_idx = np.argsort(-scores)

    results = [
        RerankDocument(
            id=int(i),
            score=float(scores[i]),
            rank=r + 1,
        )
        for r, i in enumerate(ranked_idx)
    ]

    return RerankResponse(
        model_id=ENCODER_MODEL_ID,
        results=results,
    )
