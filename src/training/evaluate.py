"""
Model Evaluation Script - Calculate metrics and identify failure cases
Required for submission: mAP, precision, recall, and 5 failure cases
"""

import os
import json
from pathlib import Path
from ultralytics import RTDETR
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def evaluate_model(
    model_path: str = "models/best.pt",
    data_yaml: str = "data/processed/data.yaml",
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    save_dir: str = "evaluation_results"
):
    """
    Comprehensive model evaluation

    Args:
        model_path: Path to trained model
        data_yaml: Dataset configuration
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        save_dir: Where to save evaluation results
    """

    print("=" * 70)
    print("Model Evaluation - Classroom Object Detection")
    print("=" * 70)

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    print(f"\n📂 Loading model: {model_path}")
    model = RTDETR(model_path)

    # Run validation
    print(f"\n🔍 Running evaluation on test set...")
    metrics = model.val(
        data=data_yaml,
        split='test',
        conf=conf_threshold,
        iou=iou_threshold,
        save_json=True,
        plots=True,
    )

    print(f"\n📊 Overall Metrics:")
    print(f"   mAP@0.5: {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")

    # Per-class metrics
    print(f"\n📈 Per-Class Metrics:")
    class_names = model.names
    for i, name in class_names.items():
        print(f"   {name}:")
        print(f"      Precision: {metrics.box.class_result(i)[0]:.4f}")
        print(f"      Recall: {metrics.box.class_result(i)[1]:.4f}")
        print(f"      mAP@0.5: {metrics.box.class_result(i)[2]:.4f}")

    # Save metrics to JSON
    metrics_dict = {
        "overall": {
            "mAP@0.5": float(metrics.box.map50),
            "mAP@0.5:0.95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "per_class": {}
    }

    for i, name in class_names.items():
        class_metrics = metrics.box.class_result(i)
        metrics_dict["per_class"][name] = {
            "precision": float(class_metrics[0]),
            "recall": float(class_metrics[1]),
            "mAP@0.5": float(class_metrics[2]),
            "mAP@0.5:0.95": float(class_metrics[3]),
        }

    metrics_file = save_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"\n💾 Metrics saved to: {metrics_file}")

    return metrics, model, class_names


def find_failure_cases(
    model,
    test_images_dir: str = "data/processed/test/images",
    num_failures: int = 5,
    save_dir: str = "evaluation_results/failures"
):
    """
    Identify and analyze failure cases

    Categories:
    1. False Negatives (missed detections)
    2. False Positives (incorrect detections)
    3. Low confidence correct detections
    4. Class confusion
    5. Bbox localization errors
    """

    print(f"\n🔍 Identifying failure cases...")
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    test_images_dir = Path(test_images_dir)
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))

    failures = []
    failure_types = defaultdict(list)

    for img_path in image_files[:50]:  # Check first 50 test images
        # Get ground truth
        label_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        # Load ground truth
        with open(label_path, 'r') as f:
            gt_boxes = [line.strip().split() for line in f.readlines()]
            gt_boxes = [[int(b[0])] + [float(x) for x in b[1:]] for b in gt_boxes]

        # Run prediction
        results = model.predict(str(img_path), conf=0.3, verbose=False)

        if len(results) == 0 or results[0].boxes is None:
            pred_boxes = []
        else:
            pred_boxes = results[0].boxes

        # Analyze failures
        # 1. Check for missed detections (false negatives)
        if len(gt_boxes) > len(pred_boxes):
            failure_types['missed_detection'].append({
                'image': str(img_path),
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes),
                'reason': f'Missed {len(gt_boxes) - len(pred_boxes)} objects'
            })

        # 2. Low confidence detections
        if len(pred_boxes) > 0:
            low_conf_dets = [b for b in pred_boxes if b.conf.item() < 0.5]
            if low_conf_dets:
                failure_types['low_confidence'].append({
                    'image': str(img_path),
                    'count': len(low_conf_dets),
                    'confidences': [float(b.conf.item()) for b in low_conf_dets]
                })

        # 3. Over-detection (false positives)
        if len(pred_boxes) > len(gt_boxes):
            failure_types['over_detection'].append({
                'image': str(img_path),
                'gt_count': len(gt_boxes),
                'pred_count': len(pred_boxes),
                'extra': len(pred_boxes) - len(gt_boxes)
            })

    # Select diverse failure cases
    print(f"\n❌ Failure Case Analysis:")

    failure_report = []

    for category, cases in failure_types.items():
        print(f"\n   {category.replace('_', ' ').title()}: {len(cases)} cases")

        if cases and len(failure_report) < num_failures:
            case = cases[0]
            failure_report.append({
                'category': category,
                'details': case
            })

            # Visualize this failure
            visualize_failure(
                model,
                case['image'],
                category,
                save_dir / f"failure_{len(failure_report)}_{category}.jpg"
            )

    # Save failure report
    report_file = save_dir.parent / "failure_cases.json"
    with open(report_file, 'w') as f:
        json.dump(failure_report, f, indent=2)

    print(f"\n💾 Failure cases saved to: {report_file}")
    print(f"   Visualizations: {save_dir}/")

    return failure_report


def visualize_failure(model, image_path, category, save_path):
    """Visualize a failure case"""

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(image_path, conf=0.3, verbose=False)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img_rgb)

    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.item()
            cls = int(box.cls.item())

            color = 'red' if conf < 0.5 else 'green'

            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            label = f"{model.names[cls]}: {conf:.2f}"
            ax.text(x1, y1 - 10, label, color=color, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(f"Failure Case: {category.replace('_', ' ').title()}", fontsize=14)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RT-DETR model")
    parser.add_argument("--model", type=str, default="models/best.pt",
                       help="Path to trained model")
    parser.add_argument("--data", type=str, default="data/processed/data.yaml",
                       help="Path to data.yaml")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold")

    args = parser.parse_args()

    # Run evaluation
    metrics, model, class_names = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        conf_threshold=args.conf
    )

    # Find failure cases
    failures = find_failure_cases(model)

    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved in: evaluation_results/")
    print(f"   - metrics.json: Detailed metrics")
    print(f"   - failure_cases.json: Failure analysis")
    print(f"   - failures/: Visualized failure cases")
