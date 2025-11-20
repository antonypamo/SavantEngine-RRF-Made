# benchmarks/run_rrf_benchmark.py
"""
RRF Benchmark helper for:
SavantEngine-RRF-Made + RRFSAVANTMADE + AGIRRFCore (Φ⁴.1∞+)

Usage patterns
--------------

1) Desde tu propio código / notebook:

    from benchmarks.run_rrf_benchmark import save_and_compare_rrf_metrics

    metrics = {
        "engine": "SavantEngine-RRF-Made",
        "encoder": "antonypamo/RRFSAVANTMADE",
        "core": "AGIRRFCore",
        "phase": "Phi4.1inf+",
        "cluster_metrics": {...},
        "retrieval_metrics": {...},
        "spectral_metrics": {...},
    }

    save_and_compare_rrf_metrics(metrics, phase="phi4_1", update_baseline=False)

2) Desde la línea de comandos (si ya tienes un JSON con métricas):

    python benchmarks/run_rrf_benchmark.py \
        --metrics-file /ruta/a/metrics.json \
        --phase phi4_1

    # Para sobreescribir el baseline con la corrida actual:
    python benchmarks/run_rrf_benchmark.py \
        --metrics-file /ruta/a/metrics.json \
        --phase phi4_1 \
        --update-baseline

El baseline se espera en:
    benchmarks/<phase>/baseline_<phase>.json
Por defecto: benchmarks/phi4_1/baseline_phi4_1.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple


# --- Paths & constants -------------------------------------------------------

REPO_ROOT = os.getcwd()
DEFAULT_PHASE = "phi4_1"


def _paths_for_phase(phase: str) -> Tuple[str, str, str]:
    """Return (bench_dir, baseline_path, current_path) for a given phase."""
    bench_dir = os.path.join(REPO_ROOT, "benchmarks", phase)
    baseline_name = f"baseline_{phase}.json"
    current_name = f"current_run_{phase}.json"
    baseline_path = os.path.join(bench_dir, baseline_name)
    current_path = os.path.join(bench_dir, current_name)
    return bench_dir, baseline_path, current_path


# --- Dataclass for key metrics -----------------------------------------------


@dataclass
class KeyMetrics:
    semantic_margin: float
    hits_at_1: float
    mrr: float
    mean_ndcg_at_3: float
    energy_ratio_rrf_neutral: float
    coherence_gap_rrf_neutral: float


# --- Helpers to derive metrics -----------------------------------------------


def _derive_key_metrics(metrics: Dict[str, Any]) -> KeyMetrics:
    """Extract the 6 core numbers from a full metrics dict."""
    cm = metrics.get("cluster_metrics", {})
    rm = metrics.get("retrieval_metrics", {})
    sm = metrics.get("spectral_metrics", {})

    def _get(path: str, source: Dict[str, Any]) -> float:
        if path not in source:
            raise KeyError(f"Missing '{path}' in metrics: {list(source.keys())}")
        return float(source[path])

    semantic_margin = _get("semantic_margin", cm)
    hits_at_1 = _get("hits_at_1", rm)
    mrr = _get("mrr", rm)
    mean_ndcg_at_3 = _get("mean_ndcg_at_3", rm)

    # energy ratio
    if "energy_ratio_rrf_neutral" in sm:
        energy_ratio = float(sm["energy_ratio_rrf_neutral"])
    else:
        e_rrf = _get("energy_mean_rrf", sm)
        e_neu = _get("energy_mean_neutral", sm)
        energy_ratio = e_rrf / e_neu if e_neu != 0.0 else float("inf")

    # coherence gap
    if "coherence_gap_rrf_neutral" in sm:
        coh_gap = float(sm["coherence_gap_rrf_neutral"])
    else:
        c_rrf = _get("coherence_mean_rrf", sm)
        c_neu = _get("coherence_mean_neutral", sm)
        coh_gap = c_rrf - c_neu

    return KeyMetrics(
        semantic_margin=semantic_margin,
        hits_at_1=hits_at_1,
        mrr=mrr,
        mean_ndcg_at_3=mean_ndcg_at_3,
        energy_ratio_rrf_neutral=energy_ratio,
        coherence_gap_rrf_neutral=coh_gap,
    )


def _augment_spectral_metrics(metrics: Dict[str, Any]) -> None:
    """
    Make sure spectral_metrics contains the derived ratio/gap fields.
    Modifies `metrics` in place.
    """
    sm = metrics.setdefault("spectral_metrics", {})
    # Energy ratio
    if "energy_ratio_rrf_neutral" not in sm:
        if "energy_mean_rrf" in sm and "energy_mean_neutral" in sm:
            e_rrf = float(sm["energy_mean_rrf"])
            e_neu = float(sm["energy_mean_neutral"])
            sm["energy_ratio_rrf_neutral"] = (
                e_rrf / e_neu if e_neu != 0.0 else float("inf")
            )
    # Coherence gap
    if "coherence_gap_rrf_neutral" not in sm:
        if "coherence_mean_rrf" in sm and "coherence_mean_neutral" in sm:
            c_rrf = float(sm["coherence_mean_rrf"])
            c_neu = float(sm["coherence_mean_neutral"])
            sm["coherence_gap_rrf_neutral"] = c_rrf - c_neu


# --- IO ----------------------------------------------------------------------


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[RRFBENCH] Saved metrics to {path}")


# --- Comparison & reporting --------------------------------------------------


def _status(delta: float, eps: float = 1e-6) -> str:
    """Return an arrow-style status for a delta."""
    if delta > eps:
        return "↑"
    if delta < -eps:
        return "↓"
    return "="


def _print_comparison_table(
    baseline: KeyMetrics, current: KeyMetrics, phase: str
) -> None:
    print("")
    print("=" * 72)
    print(f" RRF Benchmark Comparison  —  phase: {phase}")
    print("=" * 72)

    headers = ("metric", "baseline", "current", "delta", "status")
    rows = []

    def add_row(name: str, b: float, c: float):
        d = c - b
        rows.append((name, b, c, d, _status(d)))

    add_row("semantic_margin", baseline.semantic_margin, current.semantic_margin)
    add_row("hits_at_1", baseline.hits_at_1, current.hits_at_1)
    add_row("mrr", baseline.mrr, current.mrr)
    add_row("mean_ndcg_at_3", baseline.mean_ndcg_at_3, current.mean_ndcg_at_3)
    add_row(
        "energy_ratio_rrf_neutral",
        baseline.energy_ratio_rrf_neutral,
        current.energy_ratio_rrf_neutral,
    )
    add_row(
        "coherence_gap_rrf_neutral",
        baseline.coherence_gap_rrf_neutral,
        current.coherence_gap_rrf_neutral,
    )

    # Format table
    col_widths = [20, 14, 14, 14, 8]
    fmt_header = (
        f"{{:<{col_widths[0]}}} {{:>{col_widths[1]}}} {{:>{col_widths[2]}}} "
        f"{{:>{col_widths[3]}}} {{:>{col_widths[4]}}}"
    )
    fmt_row = (
        f"{{:<{col_widths[0]}}} {{:>{col_widths[1]}.6f}} {{:>{col_widths[2]}.6f}} "
        f"{{:>{col_widths[3]}.6f}} {{:>{col_widths[4]}}}"
    )

    print(fmt_header.format(*headers))
    print("-" * sum(col_widths))

    for name, b, c, d, status in rows:
        print(fmt_row.format(name, b, c, d, status))

    print("-" * sum(col_widths))
    print("Legend: ↑ improved, ↓ regressed, = unchanged")
    print("")


def compare_to_baseline(
    baseline_metrics: Dict[str, Any],
    current_metrics: Dict[str, Any],
    phase: str = DEFAULT_PHASE,
) -> None:
    """Core comparison function (assumes both dicts have the same structure)."""
    b_keys = _derive_key_metrics(baseline_metrics)
    c_keys = _derive_key_metrics(current_metrics)
    _print_comparison_table(b_keys, c_keys, phase=phase)


# --- Public API for notebooks / code -----------------------------------------


def save_and_compare_rrf_metrics(
    metrics: Dict[str, Any],
    phase: str = DEFAULT_PHASE,
    update_baseline: bool = False,
) -> None:
    """
    High-level helper:

    - Asegura que spectral_metrics tenga energy_ratio y coherence_gap.
    - Guarda la corrida actual en benchmarks/<phase>/current_run_<phase>.json
    - Si existe un baseline, lo carga y muestra comparación.
    - Si no existe baseline, opcionalmente lo crea (update_baseline=True).
    """
    bench_dir, baseline_path, current_path = _paths_for_phase(phase)

    # Aseguramos campos derivados en spectral_metrics
    _augment_spectral_metrics(metrics)

    # Guardar la corrida actual
    _save_json(metrics, current_path)

    # Baseline existente o no
    if not os.path.exists(baseline_path):
        print(f"[RRFBENCH] No baseline found at {baseline_path}")
        if update_baseline:
            print("[RRFBENCH] Creating new baseline from current metrics.")
            _save_json(metrics, baseline_path)
        else:
            print(
                "[RRFBENCH] Tip: run again with update_baseline=True to "
                "promote this run as the new baseline."
            )
        return

    # Comparar con baseline actual
    baseline_metrics = _load_json(baseline_path)
    compare_to_baseline(baseline_metrics, metrics, phase=phase)

    # Actualizar baseline si se pide explícitamente
    if update_baseline:
        print("[RRFBENCH] Updating baseline with current metrics.")
        _save_json(metrics, baseline_path)


# --- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RRF metrics against a baseline for a given phase "
            "(e.g., phi4_1)."
        )
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file with full metrics. "
            "If omitted, only comparison of existing files is possible."
        ),
    )
    parser.add_argument(
        "--phase",
        type=str,
        default=DEFAULT_PHASE,
        help="Benchmark phase name (default: phi4_1).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="If set, overwrite the baseline with the current run.",
    )

    args = parser.parse_args()
    bench_dir, baseline_path, current_path = _paths_for_phase(args.phase)

    if args.metrics_file is not None:
        # Cargar métricas del archivo proporcionado
        current_metrics = _load_json(args.metrics_file)
        _augment_spectral_metrics(current_metrics)
        _save_json(current_metrics, current_path)
    else:
        # Solo comparar baseline vs current_run si existen
        if not os.path.exists(current_path):
            raise SystemExit(
                f"No metrics file provided and no current run found at {current_path}"
            )
        current_metrics = _load_json(current_path)

    # Si no hay baseline y se pide update-baseline, crearlo ahora
    if not os.path.exists(baseline_path):
        print(f"[RRFBENCH] No baseline found at {baseline_path}")
        if args.update_baseline:
            print("[RRFBENCH] Creating new baseline from current metrics.")
            _save_json(current_metrics, baseline_path)
        else:
            print(
                "[RRFBENCH] Tip: rerun with --update-baseline to "
                "set this run as the baseline."
            )
        return

    # Comparar
    baseline_metrics = _load_json(baseline_path)
    compare_to_baseline(baseline_metrics, current_metrics, phase=args.phase)

    # Actualizar baseline desde CLI si se pide
    if args.update_baseline:
        print("[RRFBENCH] Updating baseline with current metrics.")
        _save_json(current_metrics, baseline_path)


if __name__ == "__main__":
    main()
