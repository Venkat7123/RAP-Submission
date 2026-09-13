"""
RT-DETR Training Script for Classroom Detection
Fine-tuning RT-DETR on custom classroom dataset
"""

import os
import yaml
from ultralytics import RTDETR
from datetime import datetime
import json
from pathlib import Path

def train_rtdetr(
    data_yaml: str = "data/processed/data.yaml",
    model_size: str = "rtdetr-l",  # Options: rtdetr-l, rtdetr-x
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",  # GPU device, or "cpu"
    project: str = "runs/train",
    name: str = "classroom_detection",
    patience: int = 20,  # Early stopping patience
    save_period: int = 10,  # Save checkpoint every N epochs
):
    """
    Train RT-DETR model on classroom detection dataset

    Args:
        data_yaml: Path to data configuration file
        model_size: RT-DETR model variant (rtdetr-l recommended for balance)
        epochs: Maximum training epochs
        batch_size: Batch size (reduce if OOM error)
        img_size: Input image size
        device: Device to train on (0 for GPU, cpu for CPU)
        project: Where to save training runs
        name: Experiment name
        patience: Early stopping patience
        save_period: Save checkpoint interval
    """

    print("=" * 70)
    print("RT-DETR Training for Classroom Object Detection")
    print("=" * 70)

    # Training metadata for reproducibility
    training_config = {
        "model": model_size,
        "dataset": data_yaml,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "device": device,
        "optimizer": "AdamW",
        "patience": patience,
        "timestamp": datetime.now().isoformat(),
    }

    print("\n📋 Training Configuration:")
    for key, value in training_config.items():
        print(f"   {key}: {value}")

    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"Data config not found: {data_yaml}\n"
            f"Run data preparation first: python src/utils/merge_datasets.py"
        )

    # Load and verify data config
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"\n📊 Dataset Info:")
    print(f"   Classes: {data_config['names']}")
    print(f"   Number of classes: {data_config['nc']}")
    print(f"   Train images: {data_config['train']}")
    print(f"   Val images: {data_config['val']}")

    # Initialize RT-DETR model
    print(f"\n🔧 Loading {model_size} model...")
    model = RTDETR(f"{model_size}.pt")  # Will download pretrained weights

    print(f"\n🚀 Starting training...")
    print(f"   This will take several hours for 650 images (~6-8 hours)")
    print(f"   Monitor progress: tensorboard --logdir {project}/{name}")
    print(f"   Press Ctrl+C to stop (model will be saved)\n")

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        optimizer="AdamW",
        lr0=0.0001,  # Initial learning rate
        lrf=0.01,    # Final learning rate fraction
        warmup_epochs=3,
        # Data augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation
        hsv_v=0.4,    # HSV-Value
        degrees=0.0,   # Rotation
        translate=0.1, # Translation
        scale=0.5,     # Scaling
        shear=0.0,     # Shear
        flipud=0.0,    # Flip up-down
        fliplr=0.5,    # Flip left-right
        mosaic=1.0,    # Mosaic augmentation
        mixup=0.0,     # Mixup augmentation
        # Regularization
        dropout=0.0,
        # Other settings
        verbose=True,
        seed=42,
        deterministic=True,
        workers=8,
        pretrained=True,
        close_mosaic=10,  # Disable mosaic last N epochs
    )

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)

    # Get best model path
    best_model_path = Path(project) / name / "weights" / "best.pt"
    last_model_path = Path(project) / name / "weights" / "last.pt"

    print(f"\n📁 Model Saved:")
    print(f"   Best: {best_model_path}")
    print(f"   Last: {last_model_path}")

    # Save training config alongside model
    config_save_path = Path(project) / name / "training_config.json"
    with open(config_save_path, 'w') as f:
        json.dump(training_config, f, indent=2)

    print(f"   Config: {config_save_path}")

    # Copy best model to models/ directory
    final_model_path = Path("models/best.pt")
    final_model_path.parent.mkdir(exist_ok=True)

    import shutil
    shutil.copy2(best_model_path, final_model_path)
    print(f"\n📦 Model copied to: {final_model_path}")

    print(f"\n🎯 Next Steps:")
    print(f"   1. Evaluate: python src/training/evaluate.py")
    print(f"   2. Start API: uvicorn src.api.main:app --reload")
    print(f"   3. Test API: See API_USAGE.md")

    return results, best_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RT-DETR on classroom dataset")
    parser.add_argument("--data", type=str, default="data/processed/data.yaml",
                       help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="rtdetr-l",
                       choices=["rtdetr-l", "rtdetr-x"],
                       help="Model size (rtdetr-l recommended)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size (reduce if OOM)")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--device", type=str, default="0",
                       help="Device: 0 for GPU, cpu for CPU")

    args = parser.parse_args()

    # Check for GPU
    import torch
    if args.device != "cpu" and not torch.cuda.is_available():
        print("⚠️ Warning: CUDA not available, switching to CPU")
        print("   Training on CPU will be VERY slow (24+ hours)")
        response = input("   Continue with CPU? (y/n): ")
        if response.lower() != 'y':
            print("   Please use a GPU for training or use Google Colab")
            exit(1)
        args.device = "cpu"
        args.batch = 4  # Reduce batch size for CPU

    # Train
    train_rtdetr(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
    )
