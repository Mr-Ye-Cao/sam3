#!/usr/bin/env python3
"""
SAM 3 Video Demo - Track objects across video frames
"""

import os
import sys
import io
import tempfile
import subprocess
from pathlib import Path

import requests
import numpy as np
from PIL import Image, ImageDraw
import cv2


def segment_video_frames(video_path, prompt, output_dir="video_output"):
    """Segment each frame of a video and save results"""
    import io
    import base64

    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo info: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Segmenting with prompt: '{prompt}'")
    print("-" * 50)

    frame_idx = 0
    output_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every Nth frame to speed up
        if frame_idx % 5 != 0:
            frame_idx += 1
            continue

        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Save temp image
        temp_path = f"{output_dir}/temp_frame.jpg"
        pil_image.save(temp_path)

        # Call API
        try:
            with open(temp_path, "rb") as f:
                response = requests.post(
                    "http://localhost:8000/segment/image",
                    files={"image": ("frame.jpg", f, "image/jpeg")},
                    data={"prompt": prompt, "score_threshold": 0.3}
                )

            if response.status_code == 200:
                result = response.json()
                num_det = result['num_detections']

                # Decode visualization
                vis_data = base64.b64decode(result['visualization'])
                vis_image = Image.open(io.BytesIO(vis_data))

                # Save frame
                frame_path = f"{output_dir}/frame_{frame_idx:04d}.png"
                vis_image.save(frame_path)
                output_frames.append(frame_path)

                print(f"Frame {frame_idx}: {num_det} detections")
            else:
                print(f"Frame {frame_idx}: Error - {response.status_code}")

        except Exception as e:
            print(f"Frame {frame_idx}: Error - {e}")

        frame_idx += 1

    cap.release()

    # Clean up temp file
    if os.path.exists(f"{output_dir}/temp_frame.jpg"):
        os.remove(f"{output_dir}/temp_frame.jpg")

    return output_frames, fps, width, height


def create_output_video(frame_paths, output_path, fps, width, height):
    """Create video from segmented frames"""
    print(f"\nCreating output video: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps/5, (width, height))

    for frame_path in frame_paths:
        img = cv2.imread(frame_path)
        if img is not None:
            img = cv2.resize(img, (width, height))
            out.write(img)

    out.release()
    print(f"✓ Output video saved: {output_path}")


def run_demo():
    """Run the video demo"""
    import io

    print("="*60)
    print("  🎬 SAM 3 VIDEO DEMO 🎬")
    print("="*60)

    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code != 200:
            print("❌ Server not responding!")
            return
    except:
        print("❌ Server not running! Start it with: ./start.sh")
        return

    print("✓ Server is running\n")

    # Check for existing video or create one
    video_path = None

    # Check for user's video
    for ext in ['.mp4', '.avi', '.mov', '.webm']:
        for name in ['input', 'demo', 'test', 'video']:
            path = f"{name}{ext}"
            if os.path.exists(path):
                video_path = path
                print(f"Found video: {video_path}")
                break

    if video_path is None:
        print("No video found. Creating demo from images...")
        video_path = create_demo_video_from_images()

    if video_path is None or not os.path.exists(video_path):
        print("❌ Could not create demo video")
        print("\nPlease provide a video file named 'input.mp4' or 'demo.mp4'")
        return

    # Get prompt from user or use default
    prompt = "a dog"
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])

    print(f"\n🎯 Tracking: '{prompt}'")

    # Process video
    frames, fps, width, height = segment_video_frames(
        video_path,
        prompt,
        output_dir="video_output"
    )

    if frames:
        # Create output video
        create_output_video(
            frames,
            "demo_output.mp4",
            fps,
            width,
            height
        )

        print("\n" + "="*60)
        print("  🎉 DEMO COMPLETE! 🎉")
        print("="*60)
        print(f"\nOutput files:")
        print(f"  • video_output/     - Individual frames")
        print(f"  • demo_output.mp4   - Final video")
        print(f"\nProcessed {len(frames)} frames tracking '{prompt}'")
    else:
        print("❌ No frames were processed")


if __name__ == "__main__":
    run_demo()
