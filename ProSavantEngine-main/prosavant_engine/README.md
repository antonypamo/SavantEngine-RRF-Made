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
###================================================================================================
###================================================================================================
###================================================================================================

SAVANTENGINE+12nodes=Fractal Memory 
Ejemplo de integración con Icosahedral_rrf.py

Ahora, un pequeño bridge opcional para usar tu backend icosaédrico junto al SavantEngine


import torch
import numpy as np

from prosavant_engine.savant_engine import SavantEngine, _EMBEDDER
from prosavant_engine.icosahedral_rrf import IcosahedralRRF

# 1. Instanciar SavantEngine (usa DataRepository, Ω-log, etc.)
engine = SavantEngine()

# 2. Inferir dimensión del embedder RRFSAVANTMADE
if _EMBEDDER is None:
    raise RuntimeError("No se pudo inicializar el embedder RRFSAVANTMADE (_get_embedder).")

try:
    emb_dim = _EMBEDDER.get_sentence_embedding_dimension()
except Exception:
    # fallback razonable si la API no expone esa función
    test_vec = _EMBEDDER.encode(["test"], normalize_embeddings=True)
    emb_dim = int(test_vec.shape[-1])

print(f"Embed dim = {emb_dim}")

# 3. Construir el modelo icosaédrico (12 nodos gauge + GNN subconsciente)
model_icosa = IcosahedralRRF(
    input_dim=emb_dim,
    hidden_dim=2 * emb_dim,
    output_dim=emb_dim,
)

# 4. Grafo simple de 12 nodos completamente conectado (puedes cambiarlo por tu icosaedro real)
def make_fully_connected_edge_index(num_nodes: int = 12) -> torch.Tensor:
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)

edge_index = make_fully_connected_edge_index(12)

# 5. Función helper: pasar un texto → embedding RRFSAVANTMADE → IcosahedralRRF
def run_icosa_resonance(text: str) -> np.ndarray:
    # [1, emb_dim]
    emb = _EMBEDDER.encode([text], normalize_embeddings=False)
    x = torch.from_numpy(emb).float()  # [1, emb_dim]

    # Lo levantamos a [batch, input_dim, seq_len] = [1, emb_dim, 1]
    x = x.unsqueeze(-1)

    # z = embedding latente global (puedes usar algo más sofisticado después)
    z = torch.zeros(1, emb_dim)

    with torch.no_grad():
        out = model_icosa(x, edge_index=edge_index, z=z)  # [1, emb_dim]
    return out.cpu().numpy()[0]


# 6. Ejemplo de uso conjunto
query = "qué nodo φ gobierna la ética y la coherencia en el sistema"
nodo = engine.respond(query)  # usa buscar_nodo + memoria + Ω-log
psi_icosa = run_icosa_resonance(query)

print("Respuesta SavantEngine:", nodo)
print("Vector icosaédrico ψ:", psi_icosa[:8], "...")
