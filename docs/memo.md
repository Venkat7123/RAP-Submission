# Technical Memo: Constrained Classroom Object Detection & Reasoning API

**Author**: Venkatachalam S
**Date**: September 13, 2026  
**Track**: Computer Vision + Applied ML Engineering  

---

## 1. Executive Summary
This technical memo documents the end-to-end design, implementation, and evaluation of a constrained classroom object detection model coupled with a deterministic natural language reasoning API layer. The solution utilizes a fine-tuned **RT-DETR (Real-Time Detection Transformer)** model trained on 4 classroom object classes (`board`, `chair`, `desk`, `fan`) and served via a high-performance **FastAPI** service. The API features dual endpoints: `/detect` for structured bounding box outputs and `/ask` for natural language spatial and quantitative reasoning without third-party agentic frameworks.

---

## 2. Dataset Sourcing, Composition & Split Strategy

### 2.1 Dataset Composition & Classes
The dataset was curated by combining classroom domain subsets from Roboflow Universe and custom annotations, comprising **2,377 images**:
* **`board`** (Non-COCO Class): Whiteboards, chalkboards, and interactive smartboards.
* **`chair`**: Classroom chairs and seating modules.
* **`desk`**: Student desks and teacher podiums.
* **`fan`** (Non-COCO Class): Ceiling fans and wall-mounted air circulation fans.

### 2.2 Split Strategy & Justification
* **Train Split (70%)**: 1,663 images for gradient updates.
* **Validation Split (15%)**: 357 images for hyperparameter tuning and early stopping.
* **Test Split (15%)**: 357 held-out images for final metric evaluation.
* **Justification**: A stratified 70/15/15 split maintains proportional class distribution across splits while retaining a sufficiently large test set to compute reliable mAP@0.5:0.95 metrics without data leakage.

---

## 3. Model Architecture & Fine-Tuning Parameters

| Parameter | Value | Selection Rationale |
|---|---|---|
| **Base Model** | RT-DETR-L (Ultralytics) | Transformer architecture provides global context attention for dense indoor scenes |
| **Hardware** | Kaggle 2x NVIDIA T4 GPUs | Multi-GPU data parallelism (`device=[0, 1]`) |
| **Epochs** | 30 | Optimal convergence point balancing training time (~4h) and validation loss |
| **Batch Size** | 32 (16 per GPU) | Maximizes GPU VRAM utilization while maintaining stable batch norm statistics |
| **Optimizer & LR** | AdamW (`lr0=0.0001`, `lrf=0.01`) | Warmup cosine annealing schedule for fine-tuning transformer heads |
| **Image Resolution** | 640 x 640 | Standard input size balancing speed and small object resolution (fans) |

---

## 4. Quantitative Evaluation & Performance Metrics

Evaluation on the 357 held-out test images yielded the following key metrics:

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **`board`** | 0.824 | 0.791 | 0.812 | 0.584 |
| **`chair`** | 0.841 | 0.815 | 0.835 | 0.592 |
| **`desk`** | 0.795 | 0.742 | 0.768 | 0.518 |
| **`fan`** | 0.788 | 0.725 | 0.751 | 0.495 |
| **ALL (Overall)** | **0.812** | **0.768** | **0.792** | **0.547** |

---

## 5. Root-Cause Failure Case Analysis (5 Key Failures)

1. **Flat Surface Ambiguity (Desk vs. Board)**:
   * *Observation*: Large empty wooden desk surfaces in extreme close-ups were misclassified as `board`.
   * *Root Cause*: Geometric similarity (large rectangular flat plane) when contextual cues (legs/chairs) are cropped out.
2. **Occlusion under Dense Rows**:
   * *Observation*: Partially visible chairs tucked underneath desks in dense rows had lower recall.
   * *Root Cause*: Severe bounding box overlap (>70% occlusion) causing Non-Maximum Suppression (NMS) suppression.
3. **Scale Variation for Ceiling Fans**:
   * *Observation*: Small, high-ceiling fans were occasionally undetected.
   * *Root Cause*: Resolution downsampling to 640x640 reduces 15x15 pixel ceiling fan regions to minimal feature grid cells.
4. **Lighting & Glare Reflection**:
   * *Observation*: Whiteboards with heavy sunlight reflections or bright projector glare showed lower detection confidence.
   * *Root Cause*: High luminance washes out boundary contrast between board frames and walls.
5. **Class Imbalance in Complex Scenes**:
   * *Observation*: Rooms with 30+ chairs suffered minor under-counting (detected ~24 of 30).
   * *Root Cause*: Detection transformers have a query cap (default 300 objects), but dense overlaps degrade confidence scores below threshold.

---

## 6. Part B: Reasoning Layer Architecture

### 6.1 Constraint Compliance
Per the strict technical requirements, **no agentic frameworks** (LangChain, LlamaIndex, CrewAI, AutoGen) were used. The reasoning engine in `src/api/reasoning.py` is implemented using **deterministic Python control flow** and natural language parsing rules.

### 6.2 Pipeline Logic & Intent Routing
1. **Intent Parser**: Determines whether a question requires object detection (e.g., *"How many chairs?"*) or background knowledge (e.g., *"What is a classroom?"*).
2. **Detection Integration**: Runs `/detect` to extract structured bounding box and count data.
3. **Rule-based Inference Engine**: Evaluates numerical counting, existence checks, relative comparisons (e.g., *"Are there more desks than chairs?"*), and spatial relations.
4. **Guardrail ("Insufficient Information")**: Triggers when questions ask for non-detectable properties (e.g., *"What color is the wall?"* or *"Is the room occupied?"*). Returns `confidence: "insufficient"`.

---

## 7. Deployment & Reproducibility
* **API Endpoints**: Served using FastAPI and Uvicorn (`http://127.0.0.1:8000/docs`).
* **Containerization**: Full Docker support provided via `Dockerfile` and `docker-compose.yml`.
* **Weights & Artifacts**: Weights saved at `models/best.pt` (63.2 MB).
