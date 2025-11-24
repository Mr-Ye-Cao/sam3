#!/usr/bin/env python3
"""
SAM3 Native Video Demo - Uses temporal memory tracking (NOT frame-by-frame)

This uses the proper video API with propagate_in_video for temporal consistency.
"""

import os
import sys
import torch
import numpy as np
import cv2
from tqdm import tqdm


def track_video_native(video_path, prompt, output_path="tracked_native.mp4", score_threshold=0.3):
    """
    Track objects in video using SAM3's native video API with temporal memory.

    This is different from frame-by-frame image processing - it maintains
    temporal consistency across frames using memory features.
    """
    print("=" * 60)
    print("  SAM3 NATIVE VIDEO TRACKING")
    print("=" * 60)
    print(f"\nVideo: {video_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Output: {output_path}")

    # Build video predictor
    print("\nLoading SAM3 video model...")
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor()
    # Convert model to float32 to avoid BFloat16 issues on newer GPUs
    predictor.model = predictor.model.float()

    # Clear GPU cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("Model loaded!")

    # Start session
    print(f"\nStarting video session...")
    response = predictor.handle_request({
        'type': 'start_session',
        'resource_path': video_path,
    })
    session_id = response['session_id']

    # Get video info for output
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video info: {width}x{height}, {fps:.1f} fps, {total_frames} frames")
    print(f"Session ID: {session_id[:8]}...")

    # Add text prompt on first frame
    print(f"\nAdding prompt on frame 0: '{prompt}'")
    response = predictor.handle_request({
        'type': 'add_prompt',
        'session_id': session_id,
        'frame_index': 0,
        'text': prompt,
    })

    frame_idx = response['frame_index']
    outputs = response['outputs']

    if outputs is not None:
        n_det = len(outputs.get('out_obj_ids', []))
        scores = outputs.get('out_probs', [])
        print(f"Initial detections on frame {frame_idx}: {n_det} objects")
        if len(scores) > 0:
            print(f"  Scores: {[f'{s:.1%}' for s in scores[:5]]}")
    else:
        print("No initial detections")

    # Propagate through video
    print(f"\nPropagating tracking through video...")

    # Store all frame outputs
    frame_outputs = {}

    # Forward propagation
    for result in predictor.handle_stream_request({
        'type': 'propagate_in_video',
        'session_id': session_id,
        'propagation_direction': 'forward',
        'start_frame_index': 0,
    }):
        frame_idx = result['frame_index']
        outputs = result['outputs']
        if outputs is not None:
            # Only store essential data, convert tensors to numpy to save GPU memory
            stored = {
                'out_obj_ids': outputs.get('out_obj_ids', []),
                'out_probs': outputs.get('out_probs', []),
                'out_boxes_xywh': outputs.get('out_boxes_xywh', []),
            }
            # Only store mask if it exists
            if 'out_binary_masks' in outputs and outputs['out_binary_masks'] is not None:
                masks = outputs['out_binary_masks']
                if hasattr(masks, 'cpu'):
                    masks = masks.cpu().numpy()
                stored['out_binary_masks'] = masks
            frame_outputs[frame_idx] = stored

        # Periodic GPU cleanup
        if frame_idx % 100 == 0:
            torch.cuda.empty_cache()

    print(f"Tracked {len(frame_outputs)} frames")

    # Clear GPU memory before rendering
    torch.cuda.empty_cache()

    # Create output video
    print(f"\nRendering output video...")

    # Reopen video for reading frames
    cap = cv2.VideoCapture(video_path)
    # Use H.264 codec for better compatibility (fallback to mp4v if not available)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Rendering")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply masks if we have outputs for this frame
        if frame_idx in frame_outputs:
            outputs = frame_outputs[frame_idx]
            obj_ids = outputs.get('out_obj_ids', [])
            masks = outputs.get('out_binary_masks', [])
            scores = outputs.get('out_probs', [])
            boxes = outputs.get('out_boxes_xywh', [])

            # Draw each object
            for i, (obj_id, mask, score) in enumerate(zip(obj_ids, masks, scores)):
                if score < score_threshold:
                    continue

                color = colors[int(obj_id) % len(colors)]

                # Apply mask overlay
                if mask is not None and mask.any():
                    # Ensure mask is 2D
                    if mask.ndim == 3:
                        mask = mask[0]

                    # Resize mask if needed
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask.astype(np.uint8), (width, height),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)

                    # Create colored overlay (blend frame with color)
                    color_bgr = np.array([color[2], color[1], color[0]], dtype=np.float32)
                    frame[mask] = (frame[mask].astype(np.float32) * 0.5 + color_bgr * 0.5).astype(np.uint8)

                    # Draw bounding box
                    if len(boxes) > i:
                        box = boxes[i]
                        x, y, w, h = box[0] * width, box[1] * height, box[2] * width, box[3] * height
                        x1, y1 = int(x), int(y)
                        x2, y2 = int(x + w), int(y + h)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Draw label
                        label = f"ID:{obj_id} {score:.0%}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    # Close session
    predictor.handle_request({
        'type': 'close_session',
        'session_id': session_id,
    })

    print(f"\nOutput saved: {output_path}")
    print(f"Tracked {len(frame_outputs)} frames with temporal memory")

    return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python native_video_demo.py <video_path> [prompt]")
        print("Example: python native_video_demo.py game.mp4 'little girl'")
        sys.exit(1)

    video_path = sys.argv[1]
    prompt = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "person"

    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    output_path = video_path.replace('.mp4', '_native_tracked.mp4')
    track_video_native(video_path, prompt, output_path)


if __name__ == "__main__":
    main()
