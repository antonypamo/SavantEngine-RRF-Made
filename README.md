## 🚀 Run Locally

```bash
# 1) Clone the repo
git clone https://github.com/antonypamo/SavantEngine-RRF-Made.git
cd SavantEngine-RRF-Made

# 2) (Optional but recommended) Create and activate a virtual environment
python -m venv .venv

# On Linux / macOS
source .venv/bin/activate

# On Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optional) Download and cache the base RRF model
python - << 'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("antonypamo/RRFSAVANTMADE")
EOF

# 5) Run an example or your own script
# (Adjust this to your entrypoint, e.g.)
# python savant_engine.py
