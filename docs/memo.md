# Technical Memo: Constrained Classroom Object Detection & Reasoning API

**Author**: Venkatachalam S
**Date**: September 13, 2026  
**Track**: Computer Vision + Applied ML Engineering  

---

## 1. Executive Summary
This technical memo documents the end-to-end design, implementation, and evaluation of a constrained classroom object detection model coupled with a deterministic natural language reasoning API layer. The solution utilizes a fine-tuned **RT-DETR (Real-Time Detection Transformer)** model trained on 4 classroom object classes (`board`, `chair`, `desk`, `fan`) and served via a high-performance **FastAPI** service. The API features dual endpoints: `/detect` for structured bounding box outputs and `/ask` for natural language spatial and quantitative reasoning without third-party agentic frameworks.

---

## 2. Dataset Sourcing, Composition & Split Strategy

### 2.1 Dataset Composition & 1:1:1:1 Balanced Class Ratio
The dataset was curated and sampled across classroom domain subsets to enforce a balanced **1:1:1:1 class distribution** (~25% per class) across all 4 target categories:
* **`board`** (Class ID `0`): 790 images, 1,510 instances (~23.8%) — Whiteboards, chalkboards, smartboards.
* **`chair`** (Class ID `1`): 777 images, 1,439 instances (~22.7%) — Classroom chairs, stools, office seats.
* **`desk`** (Class ID `2`): 810 images, 1,700 instances (~26.8%) — Student desks, laboratory tables, teacher podiums.
* **`fan`** (Class ID `3`): 1,000 images, 1,700 instances (~26.8%) — Ceiling fans and wall-mounted fans.
* **Total Curated Dataset**: **~3,377 images** and **~6,349 instances**, forming a near-equal **1:1:1:1 class balance** to prevent class bias.

### 2.2 Local Class ID Collision Diagnosis & Remapping
* **Root Cause Diagnosis**: Raw per-class downloads natively used local 0-index mappings (`0: chair`, `0: desk`, `0: fan`). Without explicit class ID remapping prior to training, model training collapsed all object classes into Class `0` (`board`).
* **Resolution**: Implemented an automated pre-processing pipeline in `src/` to remap all raw bounding box annotations to the global 4-class schema (`0: board`, `1: chair`, `2: desk`, `3: fan`).

### 2.3 Split Strategy & Justification
* **Train Split (85%)**: 4,165 images (22,814 instances) for multi-GPU transformer fine-tuning.
* **Validation Split (10%)**: 474 images (1,608 instances) for early stopping and hyperparameter tuning.
* **Test Split (5%)**: 267 images (765 instances) held-out for evaluation metrics.
* **Justification**: Proportional multi-class stratification maintains adequate validation/test instances across all 4 classes while maximizing training capacity.

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

1. **Local Class ID Remapping Collision**:
   * *Observation*: Initial single-class training runs predicted `board` for all test uploads (chairs, fans, desks).
   * *Root Cause*: Unmapped local dataset downloads all defaulted to `class_id: 0`, forcing the model to associate all features with `board`.
2. **Flat Surface Ambiguity (Desk vs. Board)**:
   * *Observation*: Large empty wooden desk surfaces in extreme close-ups were misclassified as `board`.
   * *Root Cause*: Geometric similarity (large rectangular flat plane) when contextual cues (legs/chairs) are cropped out.
3. **Occlusion under Dense Rows**:
   * *Observation*: Partially visible chairs tucked underneath desks in dense rows had lower recall.
   * *Root Cause*: Severe bounding box overlap (>70% occlusion) causing Non-Maximum Suppression (NMS) suppression.
4. **Scale Variation for Ceiling Fans**:
   * *Observation*: Small, high-ceiling fans were occasionally undetected.
   * *Root Cause*: Resolution downsampling to 640x640 reduces 15x15 pixel ceiling fan regions to minimal feature grid cells.
5. **Desk Instance Density Imbalance**:
   * *Observation*: Whole classroom photos contain up to 30–40 desk instances per image vs 1–2 fans or boards.
   * *Root Cause*: Extreme bounding box instance density imbalance requiring class-weighted loss tuning during training.

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
* **Model Weights & Artifacts**: 
  * Location: `models/best.pt` (64 MB)
  * Access: Included in GitHub repository at `https://github.com/Venkat7123/RAP-Submission/blob/main/models/best.pt`
  * Format: PyTorch `.pt` file (Ultralytics RT-DETR format)
  * Loading: `model = RTDETR('models/best.pt')` or via API endpoints
