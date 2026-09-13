"""
Download model from external source if not present
"""
import os
import requests
from pathlib import Path

def download_model():
    """Download model file if it doesn't exist or is corrupt"""
    model_path = Path("models/best.pt")

    # Check if model exists and is valid
    if model_path.exists() and model_path.stat().st_size > 1000000:  # > 1MB
        print(f"✅ Model exists: {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return

    print("⚠️ Model missing or corrupt, downloading...")

    # GitHub LFS download URL
    model_url = "https://media.githubusercontent.com/media/Venkat7123/RAP-Submission/main/models/best.pt"

    # Create models directory
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"📥 Downloading from: {model_url}")
        response = requests.get(model_url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        print(f"📦 File size: {total_size / 1024 / 1024:.1f} MB")

        with open(model_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (5 * 1024 * 1024) == 0:  # Every 5MB
                        print(f"  Downloaded: {downloaded / 1024 / 1024:.1f} MB")

        print(f"✅ Model downloaded successfully: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise

if __name__ == "__main__":
    download_model()
