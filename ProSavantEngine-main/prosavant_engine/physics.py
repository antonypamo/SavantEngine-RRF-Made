"""Dirac Hamiltonian dynamics used by the core system."""

from __future__ import annotations
from .geometry import IcosahedralField
from .utils import to_psi3

# Prefer the Colab helper if available; otherwise define a local fallback.
try:
    # Old notebooks/helpers may define a vector-based to_psi3 in colab_utils
    from .colab_utils import to_psi3  # type: ignore[attr-defined]
except Exception:
    import numpy as _np
    from typing import Iterable as _Iterable, Union as _Union

    _ArrayLike = _Union[_np.ndarray, _Iterable[float]]

    def to_psi3(vec: _ArrayLike) -> _np.ndarray:
        """
        Map an arbitrary 1D/2D vector into a 3D psi vector compatible with
        DiracHamiltonian.

        - If dim >= 3, take the first 3 components.
        - If dim < 3, pad with zeros.
        - If a batch is provided (2D), use the first row.
        """
        arr = _np.asarray(vec, dtype=_np.float32)

        if arr.ndim == 2:
            if arr.shape[0] == 0:
                raise ValueError("to_psi3: empty batch")
            arr = arr[0]

        if arr.ndim != 1:
            raise ValueError(f"to_psi3 expects a 1D vector or 2D batch, got {arr.shape}")

        if arr.shape[0] >= 3:
            return arr[:3].copy()

        out = _np.zeros(3, dtype=_np.float32)
        out[: arr.shape[0]] = arr
        return out

import numpy as np

class DiracHamiltonian:
    """Simplified discrete Hamiltonian operating on resonance output."""

    def __init__(self, field: IcosahedralField) -> None:
        self.field = field
        self.m = 1.0
        # Gamma metric defaults to 3×3 to match the psi vector dimensionality.
        self.gamma = np.eye(3)

    def H(self, psi) -> float:
        """Compute <psi| H |psi> with a well-defined 3×1 vector shape.

        Accepts psi as:
          - 1D (3,)        → reshaped to (3, 1)
          - 2D (1, 3)      → reshaped to (3, 1)
          - 2D (3, 1)      → kept as-is

        Anything else raises a ValueError.
        """
        psi_arr = np.asarray(psi, dtype=float)

        # --- normalize shape to (3, 1) ---
        if psi_arr.ndim == 1:
            # e.g. shape (3,)
            if psi_arr.shape[0] != 3:
                raise ValueError(
                    f"DiracHamiltonian.H expects a vector of length 3, got shape {psi_arr.shape}"
                )
            psi_col = psi_arr.reshape(3, 1)

        elif psi_arr.ndim == 2:
            # Allow (1, 3) or (3, 1)
            if psi_arr.shape == (1, 3):
                psi_col = psi_arr.reshape(3, 1)
            elif psi_arr.shape == (3, 1):
                psi_col = psi_arr
            else:
                raise ValueError(
                    "DiracHamiltonian.H expects psi with shape (3,), (1, 3) or (3, 1), "
                    f"got {psi_arr.shape}"
                )
        else:
            raise ValueError(
                f"DiracHamiltonian.H expects a 1D or 2D array, got ndim={psi_arr.ndim}"
            )

        # --- 3×3 gamma, 3×1 psi → scalar ---
        # (1, 3) @ (3, 3) @ (3, 1) → (1, 1)
        energy_mat = psi_col.T @ (self.gamma @ psi_col)
        return float(energy_mat.squeeze())

