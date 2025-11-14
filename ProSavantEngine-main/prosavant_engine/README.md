### Programmatic usage (AGIRRFCore)

You can interact with the AGI–RRF core directly from Python using the
`AGIRRFCore` facade:

```python
import bootstrap_savant  # ensures repo root is on sys.path
from prosavant_engine import AGIRRFCore

# 1) Instantiate the core (auto-wires DataRepository, geometry, Hamiltonian, etc.)
core = AGIRRFCore()

# 2) Send a query
result = core.query("Quantum resonance unification")

print("Dominant frequency:", result["dominant_frequency"])
print("Hamiltonian energy:", result["hamiltonian_energy"])
print("Coherence:", result["coherence"])
print("Φ:", result["phi"])
print("Ω:", result["omega"])

# 3) Ω-reflection summary
summary = core.omega_summary()
print("Ω summary:", summary)

# 4) Φ–Ω trajectory (Plotly figure)
fig = core.visualize_phi_omega()
fig.show()  # in notebooks, or fig.write_html("phi_omega_trajectory.html")
