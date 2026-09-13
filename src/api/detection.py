"""
Object Detection Module
Wraps RT-DETR model for inference
"""

from ultralytics import RTDETR
import cv2
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


class ObjectDetector:
    """RT-DETR Object Detector for Classroom Objects"""

    def __init__(self, model_path: str = "models/best.pt", conf_threshold: float = 0.5):
        """
        Initialize detector

        Args:
            model_path: Path to trained RT-DETR model
            conf_threshold: Confidence threshold for detections
        """

        # Download model if missing or corrupt
        if not Path(model_path).exists() or Path(model_path).stat().st_size < 1000000:
            print(f"⚠️ Model missing or corrupt at {model_path}, attempting download...")
            try:
                from download_model import download_model
                download_model()
            except ImportError:
                # If download_model.py not available, try direct download
                import requests
                model_url = "https://media.githubusercontent.com/media/Venkat7123/RAP-Submission/main/models/best.pt"
                print(f"📥 Downloading model from GitHub LFS...")
                response = requests.get(model_url, timeout=300)
                Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Model downloaded: {Path(model_path).stat().st_size / 1024 / 1024:.1f} MB")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Please train the model first or check network connection."
            )


        # Load model with CPU to reduce memory usage
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        self.model = RTDETR(model_path)
        self.model.to('cpu')  # Force CPU mode to save memory
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names

        print(f"✅ Detector loaded: {model_path}")
        print(f"   Classes: {list(self.class_names.values())}")
        print(f"   Confidence threshold: {conf_threshold}")

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float = None
    ) -> List[Dict[str, Any]]:
        """
        Run object detection on image

        Args:
            image: Input image (BGR format from cv2)
            conf_threshold: Override default confidence threshold

        Returns:
            List of detections with bbox, confidence, class
        """

        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        # Run inference with optimized NMS parameters
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=0.5,        # IoU threshold for NMS (reduce overlaps)
            max_det=300,    # Max detections per image
            verbose=False
        )

        # Parse results
        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                cls_name = self.class_names[cls_id]

                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': cls_id,
                    'class_name': cls_name
                })

        return detections

    def get_object_counts(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count objects by class

        Args:
            detections: List of detections from detect()

        Returns:
            Dictionary of {class_name: count}
        """

        counts = {}
        for det in detections:
            cls_name = det['class_name']
            counts[cls_name] = counts.get(cls_name, 0) + 1

        return counts

    def filter_by_class(
        self,
        detections: List[Dict],
        class_name: str
    ) -> List[Dict]:
        """
        Filter detections by class name

        Args:
            detections: List of detections
            class_name: Class to filter for

        Returns:
            Filtered detections
        """

        return [d for d in detections if d['class_name'] == class_name]

    def get_avg_confidence(self, detections: List[Dict]) -> float:
        """Get average confidence across all detections"""

        if not detections:
            return 0.0

        return sum(d['confidence'] for d in detections) / len(detections)


if __name__ == "__main__":
    # Test detector
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detection.py <image_path>")
        sys.exit(1)

    detector = ObjectDetector()

    img_path = sys.argv[1]
    image = cv2.imread(img_path)

    if image is None:
        print(f"Error: Could not load image: {img_path}")
        sys.exit(1)

    print(f"\n🔍 Running detection on: {img_path}")

    detections = detector.detect(image)

    print(f"\n📊 Results:")
    print(f"   Total detections: {len(detections)}")

    counts = detector.get_object_counts(detections)
    print(f"\n   Object counts:")
    for cls_name, count in counts.items():
        print(f"      {cls_name}: {count}")

    print(f"\n   Detections:")
    for i, det in enumerate(detections, 1):
        print(f"      {i}. {det['class_name']}: {det['confidence']:.2f}")
