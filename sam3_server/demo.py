#!/usr/bin/env python3
"""
SAM 3 Cool Demo
Demonstrates the power of SAM 3 with visual examples
"""

import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image
import io

def download_sample_image(url, filename):
    """Download a sample image"""
    print(f"Downloading sample image...")
    response = requests.get(url)
    response.raise_for_status()

    with open(filename, "wb") as f:
        f.write(response.content)
    print(f"✓ Saved to {filename}")
    return filename

def wait_for_server(max_attempts=30):
    """Wait for server to be ready"""
    print("Waiting for server to be ready...")
    for i in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                print("✓ Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass

        if i % 5 == 0:
            print(f"  Still waiting... ({i}/{max_attempts})")
        time.sleep(2)

    return False

def run_demo():
    """Run an awesome demo"""
    print("\n" + "="*60)
    print("  🎨 SAM 3 AWESOME DEMO 🎨")
    print("="*60 + "\n")

    # Check if server is running
    if not wait_for_server():
        print("❌ Server is not running!")
        print("\nPlease start the server first:")
        print("  ./start.sh")
        print("\nor in another terminal:")
        print("  python server.py")
        sys.exit(1)

    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Sample images (using placeholder URLs - replace with actual URLs)
    demos = [
        {
            "name": "Dog Detection",
            "url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800",
            "prompts": ["a dog", "all dogs", "the dog's face"]
        },
        {
            "name": "Street Scene",
            "url": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
            "prompts": ["a car", "all cars", "people", "trees"]
        },
        {
            "name": "People",
            "url": "https://images.unsplash.com/photo-1511632765486-a01980e01a18?w=800",
            "prompts": ["a person", "all people", "person wearing glasses"]
        }
    ]

    try:
        from client import SAM3Client
        client = SAM3Client()

        print("Running cool demonstrations...\n")

        for demo_idx, demo in enumerate(demos, 1):
            print(f"\n📸 Demo {demo_idx}/{len(demos)}: {demo['name']}")
            print("-" * 60)

            # Download sample image
            image_path = output_dir / f"demo_{demo_idx}_input.jpg"
            try:
                download_sample_image(demo['url'], image_path)
            except Exception as e:
                print(f"  ⚠ Could not download image: {e}")
                print(f"  Please manually place an image at: {image_path}")
                continue

            # Test different prompts
            for prompt_idx, prompt in enumerate(demo['prompts'], 1):
                print(f"\n  🔍 Prompt {prompt_idx}: \"{prompt}\"")
                output_path = output_dir / f"demo_{demo_idx}_prompt_{prompt_idx}.png"

                try:
                    result = client.segment_image(
                        image_path=str(image_path),
                        prompt=prompt,
                        score_threshold=0.3,
                        output_path=str(output_path)
                    )

                    print(f"     ✓ Found {result['num_detections']} objects")
                    if result['scores']:
                        avg_score = sum(result['scores']) / len(result['scores'])
                        max_score = max(result['scores'])
                        print(f"     ✓ Confidence: {max_score:.2%} (max), {avg_score:.2%} (avg)")
                    print(f"     ✓ Saved to: {output_path}")

                except Exception as e:
                    print(f"     ❌ Error: {e}")

            time.sleep(1)  # Brief pause between demos

        print("\n" + "="*60)
        print("  🎉 DEMO COMPLETE! 🎉")
        print("="*60)
        print(f"\nResults saved to: {output_dir.absolute()}")
        print("\nYou can now:")
        print("  1. Check the visualizations in demo_output/")
        print("  2. Try your own images with: python client.py image <path> <prompt>")
        print("  3. Explore the API at: http://localhost:8000/docs")

    except ImportError:
        print("❌ Could not import client module")
        sys.exit(1)

if __name__ == "__main__":
    run_demo()
