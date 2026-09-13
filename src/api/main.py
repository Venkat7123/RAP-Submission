"""
FastAPI Application - Classroom Object Detection with Reasoning
Part A: Detection endpoint
Part B: Reasoning endpoint (no frameworks - hand-written logic)
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv

# Import detection and reasoning modules
from .detection import ObjectDetector
from .reasoning import ReasoningLayer

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Classroom Object Detection API",
    description="RT-DETR-based object detection with natural language reasoning",
    version="1.0.0"
)

# Initialize models (lazy loading)
detector = None
reasoner = None

def get_detector():
    """Lazy load detector"""
    global detector
    if detector is None:
        model_path = os.getenv("MODEL_PATH", "models/best.pt")
        conf_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        detector = ObjectDetector(model_path, conf_threshold)
    return detector

def get_reasoner():
    """Lazy load reasoning layer"""
    global reasoner
    if reasoner is None:
        # API key is OPTIONAL - reasoning works with pure logic!
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or ""
        model_name = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        reasoner = ReasoningLayer(api_key, model_name)

        if not api_key:
            print("ℹ️ Reasoning layer using rule-based logic (no LLM API key provided)")
        else:
            print(f"ℹ️ Reasoning layer using {model_name} for enhanced reasoning")
    return reasoner


# Response models
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    class_id: int


class DetectionResponse(BaseModel):
    success: bool
    image_size: Dict[str, int]
    num_detections: int
    detections: List[BoundingBox]
    processing_time_ms: float


class ReasoningResponse(BaseModel):
    success: bool
    question: str
    answer: str
    confidence: str  # "high", "medium", "low", or "insufficient"
    used_detection: bool
    num_objects_analyzed: int
    processing_time_ms: float


# ==================== Part A: Detection Endpoint ====================

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(..., description="Image file (JPG, PNG)")
):
    """
    Part A: Object Detection Endpoint

    Accepts an image and returns:
    - Detected objects
    - Bounding boxes
    - Confidence scores
    - Class labels

    Example usage:
        curl -X POST "http://localhost:8000/detect" \\
             -F "file=@classroom_image.jpg"
    """

    import time
    start_time = time.time()

    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Must be an image."
            )

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Please upload a valid image file."
            )

        # Get detector
        det = get_detector()

        # Run detection
        detections = det.detect(image)

        # Format response
        bboxes = [
            BoundingBox(
                x1=float(det['bbox'][0]),
                y1=float(det['bbox'][1]),
                x2=float(det['bbox'][2]),
                y2=float(det['bbox'][3]),
                confidence=float(det['confidence']),
                class_name=det['class_name'],
                class_id=int(det['class_id'])
            )
            for det in detections
        ]

        processing_time = (time.time() - start_time) * 1000

        return DetectionResponse(
            success=True,
            image_size={"width": image.shape[1], "height": image.shape[0]},
            num_detections=len(detections),
            detections=bboxes,
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


# ==================== Part B: Reasoning Endpoint ====================

@app.post("/ask", response_model=ReasoningResponse)
async def ask_question(
    file: UploadFile = File(..., description="Image file (JPG, PNG)"),
    question: str = Form(..., description="Natural language question about the image")
):
    """
    Part B: Reasoning Layer Endpoint

    Accepts an image and a natural language question, then:
    1. Routes intent: Does this need detection or not?
    2. Calls detection if needed
    3. Reasons over structured output
    4. Returns plain language answer with confidence

    Example questions:
    - "How many chairs are in this image?"
    - "Is there a whiteboard visible?"
    - "Are there more desks or chairs?"
    - "What's the most common object?"
    - "Is anyone sitting at the desks?" (insufficient info)

    Example usage:
        curl -X POST "http://localhost:8000/ask" \\
             -F "file=@classroom.jpg" \\
             -F "question=How many chairs are visible?"
    """

    import time
    start_time = time.time()

    try:
        # Validate inputs
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}"
            )

        if not question or len(question.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image"
            )

        # Get models
        det = get_detector()
        reas = get_reasoner()

        # Process question through reasoning layer
        result = reas.process_question(
            question=question,
            image=image,
            detector=det
        )

        processing_time = (time.time() - start_time) * 1000

        return ReasoningResponse(
            success=True,
            question=question,
            answer=result['answer'],
            confidence=result['confidence'],
            used_detection=result['used_detection'],
            num_objects_analyzed=result['num_objects'],
            processing_time_ms=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning error: {str(e)}")


# ==================== Health Check ====================

@app.get("/")
async def root():
    """API root - health check"""
    return {
        "message": "Classroom Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "detect": "/detect (POST) - Object detection",
            "ask": "/ask (POST) - Natural language reasoning",
            "docs": "/docs - Interactive API documentation"
        },
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""

    model_path = os.getenv("MODEL_PATH", "models/best.pt")
    model_exists = Path(model_path).exists()

    return {
        "status": "healthy" if model_exists else "model_not_found",
        "model_path": model_path,
        "model_loaded": detector is not None,
        "reasoner_loaded": reasoner is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
