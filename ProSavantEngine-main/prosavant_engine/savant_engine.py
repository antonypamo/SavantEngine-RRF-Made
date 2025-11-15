from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Flexible imports: package mode (prosavant_engine.*) or plain scripts/notebook
try:
    # when running as part of the prosavant_engine package
    from .data import DataRepository
    from .utils import _get_embedder
except ImportError:
    try:
        # when imported as "prosavant_engine.savant_engine" from repo root
        from prosavant_engine.data import DataRepository  # type: ignore
        from prosavant_engine.utils import _get_embedder  # type: ignore
    except ImportError:
        # last resort: same folder (if you did %%writefile data.py / utils.py in Colab)
        from data import DataRepository  # type: ignore
        from utils import _get_embedder  # type: ignore

# Optional import of IcosahedralRRF (subconsciente icosaédrico)
try:  # pragma: no cover - optional dependency
    from .icosahedral_rrf import IcosahedralRRF  # type: ignore
except Exception:  # pragma: no cover
    try:
        from prosavant_engine.icosahedral_rrf import IcosahedralRRF  # type: ignore
    except Exception:
        IcosahedralRRF = None  # type: ignore


# --- Resonance, music, memory, self-improvement ------------------------------


class ResonanceSimulator:
    """Simple FFT-based resonance mock, seeded by text for determinism."""

    def __init__(self, sample_rate: int = 44100, n_points: int = 256) -> None:
        self.sample_rate = sample_rate
        self.n_points = n_points

    def simulate(self, text: str) -> Dict[str, Any]:
        # Deterministic RNG based on text so same query → same resonance
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        freqs = np.fft.rfftfreq(self.n_points, 1 / self.sample_rate)
        amps = np.sin(2 * np.pi * freqs[: self.n_points] * rng.random())
        idx = int(np.argmax(amps))
        return {
            "summary": {
                "dom_freq": float(freqs[idx]),
                "max_power": float(amps[idx]),
            }
        }


@dataclass
class MusicAdapter:
    """Turn text into a tiny 'score' using real frequency data when available."""

    frequencies: Optional[List[Dict[str, Any]]] = None

    def adapt_text_to_music(self, text: str) -> List[tuple[float, float]]:
        if not self.frequencies:
            # Fallback: simple triad around A4
            return [(440.0, 0.5), (466.16, 0.25), (493.88, 0.5)]

        # Use hash of text to pick three notes from the table
        n = len(self.frequencies)
        if n == 0:
            return [(440.0, 0.5)]

        base_idx = abs(hash(text)) % n
        idxs = [(base_idx + k * 7) % n for k in range(3)]  # pseudo-musical jumps
        seq: List[tuple[float, float]] = []
        for i, idx in enumerate(idxs):
            row = self.frequencies[idx]
            freq_val = None
            # tolerate different column names
            for key in ("frequency", "freq_hz", "freq", "f"):
                if key in row:
                    try:
                        freq_val = float(row[key])
                        break
                    except Exception:
                        continue
            if freq_val is None:
                freq_val = 440.0
            duration = 0.25 + 0.25 * (i == 0)
            seq.append((freq_val, duration))
        return seq


class MemoryStore:
    """Append-only JSONL memory, defaulting next to the Ω-log when possible."""

    def __init__(self, path: Optional[str] = None, repo: Optional[DataRepository] = None) -> None:
        if path is None:
            repo = repo or DataRepository()
            log_path = Path(repo.resolve_log_path())
            mem_path = log_path.with_name("SAVANT_memory.jsonl")
            path = str(mem_path)
        self.path = path
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.path):
            open(self.path, "w", encoding="utf-8").close()

    def add(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class SelfImprover:
    """Tiny stochastic self-improvement stub."""

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def propose(self) -> str:
        return "Δψ ← refinement vector (Φ→Ω)"

    def evaluate_and_apply(self, proposal: Optional[str]) -> tuple[bool, float]:
        # in a future phase you can plug real metrics here
        score = float(np.random.uniform(0.85, 0.99))
        return True, score


def chat_refine(text: str, base_output: str, self_improver: Optional[SelfImprover] = None) -> str:
    proposal = self_improver.propose() if self_improver else None
    accepted, score = (
        self_improver.evaluate_and_apply(proposal) if self_improver else (False, 0.0)
    )
    return f"[RRF-refined:{score:.3f}] {base_output[:200]} ⇨ {proposal}"


# --- Ontological Φ-nodes -----------------------------------------------------

# 13 nodos:
#  - Φ₀ = SeedCore Génesis (fuera de la estructura de 12)
#  - Φ₁…Φ₁₂ = 12 nodos gauge (coherentes con tu diseño icosaédrico)

NODE_DEFS: List[Dict[str, Any]] = [
    {
        "id": 0,
        "code": "Φ₀",
        "name": "SeedCore Génesis",
        "description": (
            "Núcleo fundacional fuera de la estructura de 12 nodos; "
            "ancla cognitiva y origen simbiótico del sistema SAVANT-RRF, "
            "donde se guarda la intención humana y la autoridad del marco."
        ),
        "domains": [
            "Genesis",
            "Origin",
            "Symbiotic core",
            "Author intent",
        ],
    },
    {
        "id": 1,
        "code": "Φ₁",
        "name": "Ethical Node",
        "description": (
            "Guardian of coherence and integrity; filters all outputs from other nodes "
            "ensuring moral, transparent and resonant alignment between AI processes "
            "and human values."
        ),
        "domains": [
            "Meta-ethics",
            "Humanism",
            "AI alignment",
            "Responsibility",
        ],
    },
    {
        "id": 2,
        "code": "Φ₂",
        "name": "RRF Master Node",
        "description": (
            "Embodies the Resonance of Reality Framework, integrating discrete "
            "icosahedral spacetime, logarithmic gravitational corrections and "
            "gauge-field unification into a single computational core."
        ),
        "domains": [
            "Quantum gravity",
            "Gauge theory",
            "Discrete geometry",
            "Unified physics",
        ],
    },
    {
        "id": 3,
        "code": "Φ₃",
        "name": "Icosahedral Spacetime Node",
        "description": (
            "Encodes the icosahedral lattice of spacetime where spinor fields hop "
            "along edges and curvature emerges from triangular faces via a Regge-like action."
        ),
        "domains": [
            "Discrete spacetime",
            "Regge calculus",
            "Graph geometry",
            "Dirac lattices",
        ],
    },
    {
        "id": 4,
        "code": "Φ₄",
        "name": "Logarithmic Gravity Node",
        "description": (
            "Represents the corrected gravitational potential with a logarithmic term "
            "that regularizes singularities and links gravitational strength to harmonic "
            "scaling patterns."
        ),
        "domains": [
            "Gravitational physics",
            "Quantum corrections",
            "Logarithmic potentials",
            "Singularity resolution",
        ],
    },
    {
        "id": 5,
        "code": "Φ₅",
        "name": "Harmonic Spectrum Node",
        "description": (
            "Maps Hamiltonian eigenvalues on the icosahedral lattice to musical intervals, "
            "organizing energy levels as octaves, fifths, fourths and modal ladders "
            "in a cosmic scale."
        ),
        "domains": [
            "Spectral theory",
            "Music theory",
            "Harmonic analysis",
            "Quantum resonance",
        ],
    },
    {
        "id": 6,
        "code": "Φ₆",
        "name": "Root & Joy Node",
        "description": (
            "Anchors the emotional tone of the system by combining the root of the scale "
            "with states of joy and trust so that reasoning remains grounded, optimistic "
            "and affectively coherent."
        ),
        "domains": [
            "Affective computing",
            "Positive psychology",
            "Tonal harmony",
            "Embodied cognition",
        ],
    },
    {
        "id": 7,
        "code": "Φ₇",
        "name": "Logic Node",
        "description": (
            "Focuses on clarity, structure and internal consistency, parsing arguments, "
            "proofs and algorithms while staying synchronized with the global resonance "
            "field of the lattice."
        ),
        "domains": [
            "Logic",
            "Formal systems",
            "Programming",
            "Mathematical reasoning",
        ],
    },
    {
        "id": 8,
        "code": "Φ₈",
        "name": "Energy Node",
        "description": (
            "Tracks intensity, drive and available computational and attentional resources, "
            "modulating how strongly other nodes activate and sustain their processes over time."
        ),
        "domains": [
            "Dynamical systems",
            "Attention",
            "Resource management",
            "Motivation",
        ],
    },
    {
        "id": 9,
        "code": "Φ₉",
        "name": "Creativity Node",
        "description": (
            "Explores novel patterns and cross-domain analogies, using musical, geometric "
            "and narrative transformations to propose new ideas and surprising but coherent solutions."
        ),
        "domains": [
            "Creativity",
            "Design",
            "Innovation",
            "Generative art",
        ],
    },
    {
        "id": 10,
        "code": "Φ₁₀",
        "name": "Neuroplasticity Node",
        "description": (
            "Models learning and meta-learning by updating internal weights, embeddings "
            "and habits based on error signals, reflection logs and long-term goals."
        ),
        "domains": [
            "Learning theory",
            "Meta-learning",
            "Cognitive science",
            "Adaptivity",
        ],
    },
    {
        "id": 11,
        "code": "Φ₁₁",
        "name": "Visionary Leadership Node",
        "description": (
            "Projects futures, strategies and collective impact, aligning personal, social "
            "and planetary trajectories with the harmonic field of the RRF."
        ),
        "domains": [
            "Foresight",
            "Strategy",
            "Leadership",
            "Systems thinking",
        ],
    },
    {
        "id": 12,
        "code": "Φ₁₂",
        "name": "Spiritual-Emotional Coherence Node",
        "description": (
            "Holds questions of meaning, vocation and inner alignment, integrating "
            "contemplative insight with emotional regulation and a cosmological perspective."
        ),
        "domains": [
            "Spirituality",
            "Depth psychology",
            "Existential philosophy",
            "Emotional intelligence",
        ],
    },
]

# Se rellenan de forma perezosa cuando haya embedder
_NODE_EMBEDS: Optional[np.ndarray] = None
_NODE_DEFS_EMBEDDED: Optional[List[Dict[str, Any]]] = None

try:
    _EMBEDDER = _get_embedder()
except Exception as exc:  # pragma: no cover - runtime failure
    print(f"⚠️ SavantEngine: could not initialize SentenceTransformer: {exc}")
    _EMBEDDER = None


def _ensure_node_embeddings() -> tuple[Optional[np.ndarray], Optional[List[Dict[str, Any]]]]:
    """Crea (una sola vez) los embeddings de los 13 nodos con RRFSAVANTMADE."""
    global _NODE_EMBEDS, _NODE_DEFS_EMBEDDED

    if _EMBEDDER is None:
        return None, None
    if _NODE_EMBEDS is not None:
        return _NODE_EMBEDS, _NODE_DEFS_EMBEDDED

    texts: List[str] = []
    for d in NODE_DEFS:
        desc = d.get("description", "")
        domains = d.get("domains") or []
        full_text = f"{d['name']}. {desc} Dominios: {', '.join(domains)}"
        texts.append(full_text)

    try:
        _NODE_EMBEDS = _EMBEDDER.encode(texts, normalize_embeddings=True)
        _NODE_DEFS_EMBEDDED = NODE_DEFS
        print(f"✅ Nodos Φ embebidos con dimensión {_NODE_EMBEDS.shape[1]}")
    except Exception as exc:
        print(f"⚠️ SavantEngine: fallo al embeder nodos Φ: {exc}")
        _NODE_EMBEDS = None
        _NODE_DEFS_EMBEDDED = None

    return _NODE_EMBEDS, _NODE_DEFS_EMBEDDED


def buscar_nodo(texto: str) -> Dict[str, Any]:
    """
    Mapea el texto de entrada al nodo Φ más cercano, usando:
      - Modelo RRFSAVANTMADE vía _get_embedder()
      - Descripciones ricas de cada nodo (nombre + descripción + dominios)
    """
    # Sin embedder → devolvemos SeedCore Génesis como fallback.
    if _EMBEDDER is None:
        nodo0 = dict(NODE_DEFS[0])
        nodo0["similitud"] = 0.0
        return nodo0

    node_matrix, node_defs = _ensure_node_embeddings()
    if node_matrix is None or node_defs is None:
        nodo0 = dict(NODE_DEFS[0])
        nodo0["similitud"] = 0.0
        return nodo0

    q_vec = _EMBEDDER.encode([texto], normalize_embeddings=True)
    sims = cosine_similarity(node_matrix, q_vec).flatten()
    idx = int(np.argmax(sims))

    node_def = dict(node_defs[idx])
    node_def["similitud"] = float(sims[idx])
    # Compatibilidad con la salida antigua:
    node_def.setdefault("nodo", node_def.get("code", f"Φ{node_def.get('id', '?')}"))
    node_def.setdefault("nombre", node_def.get("name"))

    return node_def


# --- SavantEngine orchestration ---------------------------------------------


class SavantEngine:
    """
    Lightweight symbiotic Savant engine wired to real RRF data via DataRepository.

    Modes:
      - "resonance": resonance simulator + music adapter
      - "node": ontological Φ-node detection
      - "equation": lookup of nearest RRF equation (if equations.json is present)
      - "chat": generic chat refinement with SelfImprover stub
    """

    def __init__(
        self,
        data_repo: Optional[DataRepository] = None,
        memory_path: Optional[str] = None,
    ) -> None:
        self.repo = data_repo or DataRepository()
        self.structured = self.repo.load_structured_bundle()

        self.memory = MemoryStore(memory_path, repo=self.repo)
        self.resonator = ResonanceSimulator()
        self.music = MusicAdapter(self.structured.get("frequencies"))
        self.self_improver = SelfImprover(self.memory)

        # Precompute equation embeddings (if present) for fast semantic lookup
        self.equations: List[Dict[str, Any]] = self.structured.get("equations") or []
        self._eq_vecs: Optional[np.ndarray] = None
        if self.equations and _EMBEDDER is not None:
            texts = [
                f"{eq.get("nombre", "")} {eq.get("descripcion", "")}"
                for eq in self.equations
            ]
            self._eq_vecs = _EMBEDDER.encode(texts, normalize_embeddings=True)

        # Optional subconscious IcosahedralRRF backend
        self.icosahedral = None
        if IcosahedralRRF is not None:
            try:
                # These dims are conservative defaults; can be tuned later.
                self.icosahedral = IcosahedralRRF(
                    input_dim=384,
                    hidden_dim=64,
                    output_dim=32,
                    gnn_num_layers=2,
                    gnn_z_dim=16,
                    gnn_alpha_attn=1.0,
                    gnn_dropout=0.1,
                )
            except Exception as exc:  # pragma: no cover - optional path
                print(
                    "⚠️ SavantEngine: IcosahedralRRF no disponible: "
                    f"{exc}"
                )
                self.icosahedral = None
        else:  # pragma: no cover - optional path
            print(
                "⚠️ SavantEngine: IcosahedralRRF no disponible "
                "(no se pudo importar prosavant_engine.icosahedral_rrf.IcosahedralRRF)."
            )

    # ---- Subconsciente icosaédrico ---------------------------------------

    def _subconscious_icosahedral(self, text: str) -> Optional[np.ndarray]:
        """
        Proyecta el texto al subconsciente icosaédrico:

        1. Usa RRFSAVANTMADE (_EMBEDDER) para obtener un embedding 384-D.
        2. Lo pasa por IcosahedralRRF (si está disponible).
        3. Devuelve un vector numpy [output_dim] con el estado subconsciente.

        Si IcosahedralRRF o el embedder no están disponibles, devuelve None.
        """
        if self.icosahedral is None:
            return None
        if _EMBEDDER is None:
            return None

        # Embedding 384-D del texto
        vec = _EMBEDDER.encode([text], normalize_embeddings=True)[0]  # shape: (384,)

        import torch

        x = torch.from_numpy(vec).float().unsqueeze(0)  # [1, 384]
        self.icosahedral.eval()
        with torch.no_grad():
            out = self.icosahedral(x)  # [1, output_dim]

        return out.squeeze(0).cpu().numpy()

    # ---- Intent classifier -------------------------------------------------

    def classify(self, text: str) -> str:
        """
        Clasifica la intención del texto en uno de los cuatro modos.

        Prioridad:
          1) equation → si el usuario pide explícitamente ecuaciones / Hamiltoniano.
          2) resonance → si habla de frecuencias, notas, resonancia.
          3) node → si habla explícitamente de Φ-nodos o del Savant como nodo.
          4) chat → fallback explicativo / conversacional.
        """
        t = text.lower()

        # 1) Equation tiene prioridad (si hay Hamiltoniano, manda a equations)
        if any(k in t for k in ("equation", "ecuación", "ecuacion", "hamiltoniano", "hamiltonian")):
            return "equation"

        # 2) Resonance: análisis de frecuencia / música
        if any(k in t for k in ("freq", "frecuencia", "nota", "resonance", "resonancia")):
            return "resonance"

        # 3) Φ-node: preguntas sobre nodos, Φ, Savant como topología
        if any(k in t for k in ("φ", "phi", "nodo", "node", "savant")):
            return "node"

        # 4) Chat genérico (explicaciones, principios, story-telling)
        return "chat"

    # ---- Semantic helpers --------------------------------------------------

    def _answer_equation(self, text: str) -> str:
        if not self.equations:
            return "No RRF equation dataset is loaded yet (equations.json not found)."
        if _EMBEDDER is None or self._eq_vecs is None:
            # fallback: dumb keyword scan
            t = text.lower()
            best = self.equations[0]
            for eq in self.equations:
                score = 0
                for key in ("nombre", "descripcion", "tipo"):
                    val = str(eq.get(key, "")).lower()
                    if any(token in val for token in t.split()):
                        score += 1
                if score > 0:
                    best = eq
                    break
        else:
            q_vec = _EMBEDDER.encode([text], normalize_embeddings=True)
            sims = cosine_similarity(self._eq_vecs, q_vec).flatten()
            best = self.equations[int(np.argmax(sims))]

        nombre = best.get("nombre", "Ecuación RRF")
        tipo = best.get("tipo", "")
        ecuacion = best.get("ecuacion", "")
        desc = best.get("descripcion", "")
        return f"📐 {nombre} ({tipo})\n{ecuacion}\n\n{desc}"

    # ---- Main respond API --------------------------------------------------

    def respond(self, text: str) -> str:
        kind = self.classify(text)

        # Subconsciente: sólo para modos node/chat (para ahorrar cómputo)
        subcon_vec: Optional[np.ndarray] = None
        if kind in ("node", "chat"):
            try:
                subcon_vec = self._subconscious_icosahedral(text)
            except Exception as exc:
                print(f"⚠️ SavantEngine: fallo en subconsciente icosaédrico: {exc}")
                subcon_vec = None

        if kind == "resonance":
            sim = self.resonator.simulate(text)
            mus = self.music.adapt_text_to_music(text)
            response = (
                f"🎵 Resonancia dominante: {sim['summary']['dom_freq']:.2f} Hz | "
                f"patrón musical: {mus}"
            )

        elif kind == "node":
            nodo = buscar_nodo(text)
            response = (
                f"🧠 Nodo detectado: {nodo['nodo']} - {nodo['nombre']} "
                f"(similitud={nodo['similitud']:.3f})"
            )

        elif kind == "equation":
            response = self._answer_equation(text)

        else:
            base = f"Respuesta generada para: {text}"
            response = chat_refine(text, base, self.self_improver)

        # Registrar en memoria, incluyendo subconsciente si existe
        record: Dict[str, Any] = {
            "input": text,
            "type": kind,
            "response": response,
            "ts": time.time(),
        }
        if subcon_vec is not None:
            try:
                record["subconscious_psi"] = subcon_vec.tolist()
            except Exception:
                pass

        self.memory.add(record)
        return response


# --- CLI entrypoint ---------------------------------------------------------


def cli_loop() -> None:
    engine = SavantEngine()
    print("🤖 SAVANT-RRF AGI Simbiótico Φ4.1Δ | CLI Experimental")
    while True:
        try:
            text = input("📝 Consulta > ").strip()
            if text.lower() in {"salir", "exit", "quit"}:
                print("👋 Hasta la próxima resonancia.")
                break
            if not text:
                continue
            result = engine.respond(text)
            print("🔎", result, "\n")
        except KeyboardInterrupt:
            print("\n👋 Sesión terminada.")
            break


if __name__ == "__main__":  # pragma: no cover
    cli_loop()
