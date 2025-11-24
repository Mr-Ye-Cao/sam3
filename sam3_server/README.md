# SAM 3 Inference Server

FastAPI server for SAM 3 segmentation on RTX 5090.

## Installation (RTX 5090)

```bash
# 1. Create conda environment
conda create -n sam3 python=3.12 -y
conda activate sam3

# 2. Install PyTorch nightly (required for RTX 5090 sm_120)
pip install --no-deps torch==2.10.0.dev20251021+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --no-deps torchvision==0.25.0.dev20251021+cu128 torchaudio==2.10.0.dev20251021+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install CUDA 12.8 dependencies
pip install nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cupti-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 \
  nvidia-cuda-runtime-cu12==12.8.90 nvidia-cufft-cu12==11.3.3.83 nvidia-curand-cu12==10.3.9.90 \
  nvidia-cusolver-cu12==11.7.3.90 nvidia-cusparse-cu12==12.5.8.93 nvidia-nccl-cu12==2.27.5 \
  nvidia-nvjitlink-cu12==12.8.93 nvidia-nvtx-cu12==12.8.90 nvidia-cufile-cu12==1.13.1.3
pip install nvidia-nvshmem-cu12==3.3.20 pytorch-triton==3.5.0+git7416ffcb --index-url https://download.pytorch.org/whl/nightly/cu128

# 4. Install SAM3 package
cd /path/to/sam3
pip install -e ".[notebooks]"

# 5. Install server dependencies
pip install fastapi uvicorn python-multipart

# 6. Authenticate with HuggingFace and download model (one-time)
# First request access at: https://huggingface.co/facebook/sam3
# Then get token from: https://huggingface.co/settings/tokens
python -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"

# 7. Download model checkpoint (auto-downloads on first run, or manually):
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt')"
# Model cached at: ~/.cache/huggingface/hub/models--facebook--sam3/
```

## Quick Start

```bash
conda activate sam3
cd sam3_server
python server.py
```

Server starts at `http://localhost:8000` (API docs: `/docs`)

## Usage

### CLI
```bash
python client.py health                                    # Health check
python client.py image photo.jpg "a dog" -o result.png    # Segment image
python client.py video video.mp4 "a car"                  # Track in video
```

### Python
```python
from client import SAM3Client

client = SAM3Client("http://localhost:8000")
result = client.segment_image("photo.jpg", "a dog", output_path="result.png")
print(f"Found {result['num_detections']} dogs")
```

### curl
```bash
curl -X POST http://localhost:8000/segment/image \
  -F "image=@photo.jpg" -F "prompt=a dog" -F "score_threshold=0.3"
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/segment/image` | POST | Segment image (params: image, prompt, score_threshold, max_detections) |
| `/segment/video/start` | POST | Start video session (params: video) |
| `/segment/video/add_prompt` | POST | Add prompt to video (params: session_id, frame_index, text) |

### Response Format (image)
```json
{
  "num_detections": 2,
  "boxes": [[x1, y1, x2, y2], ...],
  "scores": [0.95, 0.87],
  "visualization": "base64_png"
}
```

## Text Prompt Examples

- Simple: `"a dog"`, `"a car"`, `"a person"`
- Attributes: `"a red car"`, `"person wearing blue"`
- Multiple: `"all dogs"`, `"people playing soccer"`
- Specific: `"the dog's nose"`, `"person holding a phone"`

## Troubleshooting

```bash
# Check GPU
nvidia-smi

# Test CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Test SAM3 imports
python -c "from sam3.model_builder import build_sam3_image_model"

# Re-login to HuggingFace
python -m huggingface_hub.cli.cli login

# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"
```

## Files

- `server.py` - FastAPI server
- `client.py` - Python client + CLI
- `start.sh` - Startup script
- `demo.py` - Demo script
