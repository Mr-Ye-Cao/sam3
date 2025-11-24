#!/usr/bin/env python3
"""
SAM 3 Client
Easy-to-use client for SAM 3 Inference API
"""

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Optional

import requests
from PIL import Image


class SAM3Client:
    """Client for SAM 3 Inference API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health_check(self):
        """Check if the server is healthy"""
        response = requests.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()

    def segment_image(
        self,
        image_path: str,
        prompt: str,
        score_threshold: float = 0.3,
        max_detections: int = 200,
        output_path: Optional[str] = None
    ):
        """
        Segment objects in an image

        Args:
            image_path: Path to image file
            prompt: Text description of what to segment
            score_threshold: Minimum confidence score (0-1)
            max_detections: Maximum number of objects to detect
            output_path: Optional path to save visualization

        Returns:
            Dict with segmentation results
        """
        with open(image_path, "rb") as f:
            files = {"image": (Path(image_path).name, f, "image/jpeg")}
            data = {
                "prompt": prompt,
                "score_threshold": score_threshold,
                "max_detections": max_detections
            }

            response = requests.post(
                f"{self.base_url}/segment/image",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()

        # Save visualization if requested
        if output_path and "visualization" in result:
            vis_data = base64.b64decode(result["visualization"])
            Image.open(io.BytesIO(vis_data)).save(output_path)
            print(f"Saved visualization to {output_path}")

        return result

    def start_video_session(self, video_path: str):
        """Start a video segmentation session"""
        with open(video_path, "rb") as f:
            files = {"video": (Path(video_path).name, f, "video/mp4")}
            response = requests.post(
                f"{self.base_url}/segment/video/start",
                files=files
            )
            response.raise_for_status()
            return response.json()

    def add_video_prompt(
        self,
        session_id: str,
        text: str,
        frame_index: int = 0
    ):
        """Add a text prompt to a video session"""
        data = {
            "session_id": session_id,
            "frame_index": frame_index,
            "text": text
        }

        response = requests.post(
            f"{self.base_url}/segment/video/add_prompt",
            data=data
        )
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3 Client - Segment anything with text prompts"
    )
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Health check command
    subparsers.add_parser("health", help="Check server health")

    # Image segmentation command
    image_parser = subparsers.add_parser("image", help="Segment an image")
    image_parser.add_argument("image", help="Path to image file")
    image_parser.add_argument("prompt", help="What to segment (e.g., 'a person')")
    image_parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Minimum confidence score (default: 0.3)"
    )
    image_parser.add_argument(
        "--max-detections",
        type=int,
        default=200,
        help="Maximum number of detections (default: 200)"
    )
    image_parser.add_argument(
        "-o", "--output",
        help="Output path for visualization"
    )

    # Video segmentation command
    video_parser = subparsers.add_parser("video", help="Segment a video")
    video_parser.add_argument("video", help="Path to video file")
    video_parser.add_argument("prompt", help="What to track (e.g., 'a person')")
    video_parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to apply prompt (default: 0)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = SAM3Client(args.server)

    try:
        if args.command == "health":
            result = client.health_check()
            print(json.dumps(result, indent=2))

        elif args.command == "image":
            result = client.segment_image(
                image_path=args.image,
                prompt=args.prompt,
                score_threshold=args.score_threshold,
                max_detections=args.max_detections,
                output_path=args.output
            )

            print(f"\nFound {result['num_detections']} objects matching '{args.prompt}'")
            print(f"\nScores:")
            for i, score in enumerate(result['scores'], 1):
                print(f"  {i}. {score:.3f}")

            if args.output:
                print(f"\nVisualization saved to: {args.output}")

        elif args.command == "video":
            print(f"Starting video session...")
            session = client.start_video_session(args.video)
            print(f"Session ID: {session['session_id']}")

            print(f"Adding prompt '{args.prompt}'...")
            result = client.add_video_prompt(
                session_id=session['session_id'],
                text=args.prompt,
                frame_index=args.frame
            )

            print(f"\nProcessed {result['num_frames']} frames")
            print(f"Total detections: {len(result['detections'])}")

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {args.server}")
        print("Make sure the server is running with: python sam3_server/server.py")
        sys.exit(1)

    except requests.exceptions.HTTPError as e:
        print(f"Error: {e}")
        print(f"Response: {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    import io
    main()
