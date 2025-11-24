#!/usr/bin/env python3
"""
SAM 3 Inference Server
A FastAPI-based server for SAM 3 image and video segmentation.
"""

import io
import base64
import tempfile
import os
from pathlib import Path
from typing import Optional, List
import logging

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
from sam3.model.sam3_image_processor import Sam3Processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SAM 3 Inference API",
    description="High-performance API for SAM 3 segmentation on images and videos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
image_model = None
image_processor = None
video_predictor = None


class ImageSegmentRequest(BaseModel):
    """Request model for image segmentation"""
    prompt: str
    max_detections: Optional[int] = 200
    score_threshold: Optional[float] = 0.3


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: dict
    device: str
    cuda_available: bool


@app.on_event("startup")
async def load_models():
    """Load SAM 3 models on startup"""
    global image_model, image_processor, video_predictor

    logger.info("Loading SAM 3 models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    try:
        # Load image model
        logger.info("Loading image model...")
        image_model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            load_from_HF=True
        )
        # Convert to float32 for compatibility with nightly PyTorch
        image_model = image_model.float()
        image_processor = Sam3Processor(image_model)
        logger.info("✓ Image model loaded successfully")

        # Load video predictor
        logger.info("Loading video predictor...")
        video_predictor = build_sam3_video_predictor()
        logger.info("✓ Video predictor loaded successfully")

        logger.info("All models loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.error("Make sure you've authenticated with HuggingFace:")
        logger.error("  1. Request access at https://huggingface.co/facebook/sam3")
        logger.error("  2. Run: huggingface-cli login")
        raise


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "image": image_model is not None,
            "video": video_predictor is not None
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/segment/image")
async def segment_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    score_threshold: float = Form(0.3),
    max_detections: int = Form(200)
):
    """
    Segment objects in an image using text prompts

    Args:
        image: Image file (JPEG, PNG, etc.)
        prompt: Text description of what to segment (e.g., "a person", "all cars")
        score_threshold: Minimum confidence score (0-1)
        max_detections: Maximum number of objects to detect

    Returns:
        JSON with masks, boxes, scores, and visualization
    """
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    try:
        # Read and process image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

        logger.info(f"Processing image {image.filename} with prompt: '{prompt}'")

        # Run inference with float32 to avoid BFloat16 issues on newer GPUs
        with torch.amp.autocast(device_type='cuda', enabled=False):
            inference_state = image_processor.set_image(pil_image)
            output = image_processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

        # Filter by score threshold
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        # Convert to CPU and numpy
        masks_np = masks.cpu().numpy() if torch.is_tensor(masks) else masks
        boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else scores

        # Filter by score
        mask_filter = scores_np > score_threshold
        masks_filtered = masks_np[mask_filter][:max_detections]
        boxes_filtered = boxes_np[mask_filter][:max_detections]
        scores_filtered = scores_np[mask_filter][:max_detections]

        # Create visualization
        vis_image = visualize_masks(
            np.array(pil_image),
            masks_filtered,
            boxes_filtered,
            scores_filtered
        )

        # Convert visualization to base64
        vis_buffer = io.BytesIO()
        vis_image.save(vis_buffer, format="PNG")
        vis_base64 = base64.b64encode(vis_buffer.getvalue()).decode()

        logger.info(f"Found {len(masks_filtered)} objects matching '{prompt}'")

        return {
            "num_detections": int(len(masks_filtered)),
            "prompt": prompt,
            "boxes": boxes_filtered.tolist(),
            "scores": scores_filtered.tolist(),
            "visualization": vis_base64,
            "image_size": {
                "width": pil_image.width,
                "height": pil_image.height
            }
        }

    except Exception as e:
        logger.error(f"Error during image segmentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment/video/start")
async def start_video_session(
    video: UploadFile = File(...),
):
    """
    Start a video segmentation session

    Args:
        video: Video file or folder of JPEG frames

    Returns:
        Session ID for subsequent requests
    """
    if video_predictor is None:
        raise HTTPException(status_code=503, detail="Video model not loaded")

    try:
        # Save video to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as tmp:
            contents = await video.read()
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(f"Starting video session for {video.filename}")

        # Start session
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=tmp_path,
            )
        )

        session_id = response["session_id"]
        logger.info(f"Created video session: {session_id}")

        return {
            "session_id": session_id,
            "video_path": tmp_path,
            "num_frames": response.get("num_frames", -1)
        }

    except Exception as e:
        logger.error(f"Error starting video session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment/video/add_prompt")
async def add_video_prompt(
    session_id: str = Form(...),
    frame_index: int = Form(0),
    text: str = Form(...),
):
    """
    Add a text prompt to a video session

    Args:
        session_id: Session ID from start_video_session
        frame_index: Frame index to apply prompt (default: 0)
        text: Text description of what to track

    Returns:
        Segmentation results for all frames
    """
    if video_predictor is None:
        raise HTTPException(status_code=503, detail="Video model not loaded")

    try:
        logger.info(f"Adding prompt '{text}' to session {session_id} at frame {frame_index}")

        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=text,
            )
        )

        outputs = response["outputs"]

        # Convert outputs to serializable format
        result = {
            "session_id": session_id,
            "prompt": text,
            "frame_index": frame_index,
            "num_frames": len(outputs),
            "detections": []
        }

        for frame_idx, frame_output in enumerate(outputs):
            if frame_output:
                result["detections"].append({
                    "frame": frame_idx,
                    "num_objects": len(frame_output.get("masks", []))
                })

        logger.info(f"Processed {len(outputs)} frames")

        return result

    except Exception as e:
        logger.error(f"Error adding video prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def visualize_masks(image, masks, boxes, scores):
    """Visualize masks on image"""
    from PIL import ImageDraw, ImageFont

    pil_image = Image.fromarray(image.astype(np.uint8))
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Color palette
    colors = [
        (255, 0, 0, 100),    # Red
        (0, 255, 0, 100),    # Green
        (0, 0, 255, 100),    # Blue
        (255, 255, 0, 100),  # Yellow
        (255, 0, 255, 100),  # Magenta
        (0, 255, 255, 100),  # Cyan
    ]

    for idx, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        color = colors[idx % len(colors)]

        # Draw mask
        if mask.ndim == 3:
            mask = mask[0]

        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        colored_mask = Image.new("RGBA", pil_image.size, color)
        overlay.paste(colored_mask, (0, 0), mask_image)

        # Draw box
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color[:3] + (255,), width=3)

        # Draw score
        score_text = f"{score:.2f}"
        draw.text((x1, y1 - 20), score_text, fill=color[:3] + (255,))

    # Composite
    pil_image = pil_image.convert("RGBA")
    result = Image.alpha_composite(pil_image, overlay)
    return result.convert("RGB")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting SAM 3 Inference Server on {host}:{port}")
    logger.info(f"API docs will be available at http://{host}:{port}/docs")

    uvicorn.run(app, host=host, port=port)
