"""
Hugging Face Spaces Entry Point
Classroom Object Detection API with Reasoning Layer
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import FastAPI app
from src.api.main import app

# HF Spaces will automatically detect and run this FastAPI app
# No need for uvicorn.run() - Spaces handles it automatically

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
