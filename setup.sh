#!/bin/bash

# Exit if any command fails
set -e

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip inside the venv
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Optional: Confirm success
echo "✅ All dependencies installed in .venv"