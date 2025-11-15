"""Shared helper utilities for lightweight tensor munging."""

from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike


def _get_embedder() -> Optional[object]:
    """Inicializa y devuelve un modelo SentenceTransformer si está disponible.

    Prioridad de modelos:

    1. antonypamo/RRFSAVANTMADE  → embedder RRF especializado.
    2. antonypamo/ProSavantEngine → checkpoint AGI–RRF general.
    3. sentence-transformers/all-MiniLM-L6-v2 → fallback genérico.

    Si nada carga, devuelve None y el sistema usa fallbacks simbólicos.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "⚠️ _get_embedder: 'sentence-transformers' no está instalado. "
            "Instala con: pip install sentence-transformers"
        )
        return None

    model_candidates = [
        "antonypamo/RRFSAVANTMADE",          # 1️⃣ embedder RRF especializado
        "antonypamo/ProSavantEngine",        # 2️⃣ checkpoint AGI–RRF general
        "sentence-transformers/all-MiniLM-L6-v2",  # 3️⃣ fallback público
    ]

    last_error: Optional[Exception] = None

    for model_name in model_candidates:
        try:
            print(f"🔄 _get_embedder: intentando cargar '{model_name}'...")
            embedder = SentenceTransformer(model_name)
            print(f"✅ _get_embedder: modelo '{model_name}' cargado correctamente.")
            return embedder
        except Exception as e:
            print(f"⚠️ _get_embedder: no se pudo cargar '{model_name}': {e}")
            last_error = e

    print(
        "⚠️ _get_embedder: no se pudo cargar ningún modelo de la lista. "
        f"Último error: {last_error}. Las características semánticas estarán limitadas."
    )
    return None


def _hash_to_unit_vector(text: str) -> np.ndarray:
    """Hash determinista de string → vector 3D en [0, 1]."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    ints = np.frombuffer(digest, dtype=np.uint32)
    vec = ints[:3].astype(np.float64)
    max_uint = np.iinfo(np.uint32).max
    if max_uint:
        vec /= max_uint
    return vec


def to_psi3(value: ArrayLike | Iterable[float] | float | int | str | None) -> np.ndarray:
    """Map arbitrary inputs into a 3-component numpy vector.

    - Strings → vector hash 3D determinista.
    - Escalares → broadcast a 3 componentes.
    - Arrays largos → se truncan a 3.
    - Arrays cortos → se rellenan con ceros.
    """

    if isinstance(value, str):
        arr = _hash_to_unit_vector(value)
    elif value is None:
        arr = np.zeros(3, dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64).ravel()

    if arr.size == 0:
        arr = np.zeros(3, dtype=np.float64)
    elif arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant")
    elif arr.size > 3:
        arr = arr[:3]

    return arr.astype(np.float64, copy=False)


__all__ = ["to_psi3", "_get_embedder"]
