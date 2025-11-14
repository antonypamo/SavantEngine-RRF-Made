# run_core_cli.py
"""
Interactive CLI to run Prosavant Engine in 'core' mode.

Usage (from repo root):
    python run_core_cli.py

Usage (from Colab anywhere):
    !python /content/SavantEngine-RRF-Made/run_core_cli.py
"""

from __future__ import annotations

import bootstrap_savant  # noqa: F401  # ensures repo root is on sys.path

from prosavant_engine import AGIRRFCore  # imported via __init__.py


def main() -> None:
    core = AGIRRFCore()
    print("🤖 ProSavantEngine – AGI RRF Core Mode")
    print("   Type text to query the core; 'exit', 'quit' or 'salir' to leave.\n")

    while True:
        try:
            text = input("📝 Query > ").strip()
        except EOFError:
            break

        if text.lower() in {"exit", "quit", "salir"}:
            print("👋 Closing core mode.")
            break
        if not text:
            continue

        result = core.query(text)
        print(
            "🔎 Result:\n"
            f"  dominant_frequency: {result['dominant_frequency']:.3f} Hz\n"
            f"  hamiltonian_energy: {result['hamiltonian_energy']:.6e}\n"
            f"  coherence:          {result['coherence']:.4f}\n"
            f"  Φ:                  {result['phi']:.4f}\n"
            f"  Ω:                  {result['omega']:.4f}\n"
        )


if __name__ == "__main__":
    main()
