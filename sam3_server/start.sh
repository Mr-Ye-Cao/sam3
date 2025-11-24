#!/bin/bash
# SAM 3 Server Startup Script

set -e

echo "🚀 Starting SAM 3 Inference Server"
echo "=================================="

# Activate conda environment
echo "Activating sam3 environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam3

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ CUDA not found, will use CPU"
fi

# Check authentication
echo ""
echo "Checking Hugging Face authentication..."
python -c "
from huggingface_hub import whoami
try:
    user = whoami()
    print(f\"✓ Logged in as: {user['name']}\")
except:
    print('⚠ Not logged in to Hugging Face')
    print('  Run: python -m huggingface_hub.cli.cli login')
    print('  After requesting access at: https://huggingface.co/facebook/sam3')
" || true

echo ""
echo "Starting server..."
echo "API docs: http://localhost:8000/docs"
echo "Health check: http://localhost:8000/"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start server
python server.py
