---
license: other
title: Savant RRF Φ12.0 – Dirac-Resonant Conceptual Quality API
sdk: docker
emoji: 🐢
colorFrom: red
colorTo: green
pinned: true
short_description: API de evaluación conceptual resonante para LLM
---
🧠 Savant RRF Φ12.0 — Meta-Logic & Rerank API

Savant RRF Φ12.0 is a production-ready FastAPI service that exposes:

A meta-logic quality evaluator based on the Resonance of Reality Framework (RRF)

A batched semantic reranker using a custom icosahedral-resonant embedding model

A deterministic Φ-node ontology mapping layer

The system combines SentenceTransformer embeddings, spectral / resonance features, and a 15-dimensional meta-logit classifier to evaluate reasoning quality and rank documents efficiently.

🔗 Live API
Base URL: https://antonypamo-apisavant2.hf.space

📦 Models Used
Component	Model
Embedder	antonypamo/RRFSAVANTMADE
Meta-Logic	antonypamo/RRFSavantMetaLogicV2/logreg_rrf_savant.joblib
Feature Dim	15 features
Runtime	CPU (GPU optional if available)
🧩 Φ-Node Ontology

The system maps inputs to one of 8 deterministic Φ-nodes:

Index	Φ Node
0	Φ0_seed
1	Φ1_geometric
2	Φ2_gauge_dirac
3	Φ3_log_gravity
4	Φ4_resonance
5	Φ5_memory_symbiosis
6	Φ6_alignment
7	Φ7_meta_agi

This mapping is rule-based and reproducible, derived from spectral coherence, energy, and phase features.

🚀 Endpoints Overview
GET /

Root discovery endpoint.

{
  "status": "ok",
  "model": "RRFSavantMetaLogicV2",
  "version": "Φ12.0",
  "docs": "/docs",
  "endpoints": ["/manifest", "/health", "/evaluate", "/quality", "/v1/rerank"]
}

GET /health

Lightweight health check.

{ "status": "ok" }

GET /manifest

Static manifest describing the model.

{
  "model": "RRFSavantMetaLogicV2",
  "version": "Φ12.0",
  "encoder": "antonypamo/RRFSAVANTMADE",
  "features": 15,
  "phi_nodes": [...]
}

🧪 Quality Evaluation API
POST /evaluate

Evaluates the conceptual quality of a (prompt, answer) pair.

Request
{
  "prompt": "Explain what a smoke test is",
  "answer": "A smoke test is a minimal validation..."
}

Response
{
  "p_good": 0.87,
  "scores": {
    "SRRF": 0.87,
    "CRRF": 0.79,
    "E_phi": 0.82
  },
  "features": {
    "phi": 0.61,
    "omega": 0.22,
    "coherence": 0.83,
    "S_RRF": 0.81,
    "C_RRF": 0.77,
    "hamiltonian_energy": 48.3,
    "dominant_frequency": 12
  },
  "phi_node": "Φ4_resonance"
}

POST /quality

Alias of /evaluate. Same input, same output.

🔍 Semantic Reranking API
POST /v1/rerank

Ranks documents by semantic similarity to a query using batched embedding inference.

Request
{
  "query": "What is a smoke test?",
  "documents": [
    "A smoke test is a minimal system check",
    "Load tests measure concurrency",
    "Benchmarks compare systems"
  ],
  "alpha": 0.2
}


alpha is reserved for future hybrid scoring (currently unused).

Response
{
  "model_id": "antonypamo/RRFSAVANTMADE",
  "results": [
    { "id": 0, "score": 0.92, "rank": 1 },
    { "id": 2, "score": 0.51, "rank": 2 },
    { "id": 1, "score": 0.21, "rank": 3 }
  ]
}

⚙️ Runtime Constraints
Parameter	Limit
Max prompt length	8,000 chars
Max answer length	12,000 chars
Max documents	50
Max document size	6,000 chars

Payload violations return HTTP 413.

🧠 Feature Vector (15D)

The meta-logic classifier consumes:

Spectral / resonance features:

phi, omega, coherence, S_RRF, C_RRF

hamiltonian_energy, dominant_frequency

Φ-node one-hot encoding (8 dimensions)

Total = 15 features

🛠 Running Locally
pip install fastapi uvicorn sentence-transformers huggingface_hub joblib numpy

export HF_TOKEN=your_token_here
uvicorn app:app --host 0.0.0.0 --port 8000


Open:

http://127.0.0.1:8000/docs

📈 Performance Notes

Optimized for batched inference on /v1/rerank

Stable under load (0% error rate in benchmarks)

CPU-based by default; GPU reduces latency significantly

Tail latency (p95/p99) depends on concurrency and hardware

🧩 Design Philosophy

Savant RRF is not a generic classifier.

It encodes:

Discrete resonance physics

Icosahedral symbolic structure

Deterministic ontology mapping

Meta-logic scoring beyond surface semantics

This makes it suitable for:

AI evaluation & judging

RAG reranking

Cognitive profiling

Research-grade reasoning analysis

📄 License & Attribution

© 2025 Antony Padilla Morales
Resonance of Reality Framework (RRF)