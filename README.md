# SAM3 Local Project

Локальный проект с исходным кодом SAM3 и правками для запуска на Mac.

## Что внутри
- локальная папка `sam3/` с исходным кодом
- `test_sam3_hf.py` — пример запуска
- правки для совместимости с Mac/CPU

## Установка

```bash
git clone https://github.com/ArseniiShalamov/sam3-hf-clean.git
cd sam3-hf-clean

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
