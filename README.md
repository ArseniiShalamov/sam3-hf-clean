# SAM3 Local Project

Local project with SAM3 source code and modifications to run on Mac (CPU).

## Contents
- local `sam3/` folder with full source code
- `test_sam3_hf.py` — example script to run the model
- compatibility fixes for Mac / CPU environment

## Installation

```bash
git clone https://github.com/ArseniiShalamov/sam3-hf-clean.git
cd sam3-hf-clean

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
