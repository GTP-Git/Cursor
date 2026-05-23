#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

if [[ "$(uname -s)" == "Darwin" ]] && ! brew list libomp &>/dev/null; then
  echo "Installing OpenMP (required for XGBoost / LightGBM on macOS)..."
  brew install libomp
fi

echo "Installing Playwright Chromium (native arch)..."
python -m playwright install chromium

echo ""
echo "Setup complete. Start the app with:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
