# 🎓 Classroom Object Detection & Reasoning API

**RT-DETR-based Object Detection with Natural Language Reasoning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Deploy: Railway](https://img.shields.io/badge/Deploy-Railway-blueviolet)](https://rap-submission-production.up.railway.app/docs)

**Domain**: Classroom/Educational Space Detection  
**Classes**: `board`, `chair`, `desk`, `fan` (4 classes)  
**Model**: RT-DETR-L (Real-Time Detection Transformer)  
**Framework**: FastAPI + Ultralytics + PyTorch

**🌐 Live Demo**: [Railway Deployment](https://rap-submission-production.up.railway.app) *(Update with your URL)*

---

## 🎯 Features

- ✅ **Object Detection**: RT-DETR model fine-tuned on 3,377 balanced classroom images
- ✅ **Natural Language Reasoning**: Ask questions about detected objects in plain English
- ✅ **No Frameworks**: Pure Python reasoning layer (no LangChain/CrewAI/AutoGen)
- ✅ **Auto Model Download**: Automatically downloads model from GitHub LFS on first run
- ✅ **Production Ready**: FastAPI + Docker + Railway deployment

## Project Structure
```
RAP/
├── data/
│   ├── raw/              # Raw downloaded dataset
│   ├── processed/        # Preprocessed data
│   ├── train/           # Training split
│   ├── val/             # Validation split
│   └── test/            # Test split
├── src/
│   ├── training/        # Training scripts
│   ├── api/             # FastAPI application
│   └── utils/           # Utility functions
├── models/              # Trained model weights
├── notebooks/           # Jupyter notebooks for EDA
├── docs/                # Documentation and memo
└── requirements.txt     # Dependencies
```

## Recommended Datasets

### Option 1: Roboflow Universe (Manual Download Required)
Visit these links and download:
1. **Classroom Object Detection**: Search "classroom" on universe.roboflow.com
2. **Furniture Detection**: Look for indoor furniture datasets
3. **Custom combination**: Combine multiple datasets with relevant classes

### Option 2: Create Custom Dataset
- Collect classroom images from:
  - Google Images (with proper filtering)
  - Educational institution websites (public images)
  - Stock photo sites (free licenses)
- Label using: Roboflow Annotate, LabelImg, or CVAT

### Option 3: Combine Public Datasets
- COCO subset (chair, table classes)
- Custom scraped/labeled (fan, AC, whiteboard)

## Classes
1. **board** - Whiteboards/blackboards (⭐ Non-COCO)
2. **chair** - Common classroom seating
3. **desk** - Student desks and teacher podiums
4. **fan** - Ceiling or wall-mounted fans (⭐ Non-COCO)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 4GB RAM minimum (8GB recommended)
- GPU optional (works on CPU)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/Venkat7123/RAP-Submission.git
cd RAP-Submission

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file (optional for local development)
cat > .env << EOF
MODEL_PATH=models/best.pt
CONFIDENCE_THRESHOLD=0.5
HOST=0.0.0.0
PORT=7860
EOF

# 4. Start API server (model auto-downloads on first run)
python app.py
```

**Note**: Model weights (64MB) will be automatically downloaded from GitHub LFS on first startup if not present locally.

Server runs at: `http://localhost:7860`

### Docker Deployment

```bash
# Build image
docker build -t classroom-api .

# Run container
docker run -p 8000:8000 classroom-api
```

---

## 📖 API Documentation

**Interactive Docs**: Visit `http://localhost:7860/docs`

See [API_USAGE.md](API_USAGE.md) for detailed examples and sample requests.

## API Endpoints

### Part A: Detection
```
POST /detect
- Input: image file
- Output: JSON with bounding boxes, classes, confidence scores
```

### Part B: Reasoning
```
POST /ask
- Input: image + natural language question
- Output: Plain language answer with confidence
```

---

## 📊 Model Performance

| Metric | Overall Score |
|--------|---------------|
| **mAP@0.5** | 39.5% |
| **mAP@0.5:0.95** | 30.0% |
| **Precision** | 45.5% |
| **Recall** | 79.3% |

**Per-Class Performance**:
- Board: P=0.429, R=0.744, mAP@0.5=34.2%
- Chair: P=0.375, R=0.667, mAP@0.5=30.7%
- Desk: P=0.734, R=0.763, mAP@0.5=61.6% ⭐ (Best)
- Fan: P=0.282, R=1.000, mAP@0.5=31.3% (Perfect recall)

**Key Observations**:
- High recall (79.3%) indicates good detection coverage
- Moderate precision (45.5%) suggests some false positives
- Model trained for 30 epochs on balanced 1:1:1:1 dataset

See [docs/memo.md](docs/memo.md) for detailed evaluation and 5 failure case analyses.

---

## 🏗️ Training Your Own Model

```bash
# 1. Prepare dataset (see Dataset Guide above)
# 2. Run training script
python src/training/train.py

# 3. Evaluate model
python src/training/evaluate.py

# 4. Model weights saved to: models/best.pt
```

**Training Details**:
- Base Model: RT-DETR-L (Ultralytics)
- Hardware: Kaggle 2x NVIDIA T4 GPUs
- Epochs: 30
- Batch Size: 32 (16 per GPU)
- Training Time: ~4 hours
- Dataset: 3,377 images (1:1:1:1 class balance)
- Optimizer: AdamW (lr0=0.0001, lrf=0.01)
- Image Size: 640×640

---

## 📂 Project Structure

```
RAP/
├── app.py                 # FastAPI entry point
├── src/
│   ├── api/
│   │   ├── main.py       # FastAPI application
│   │   ├── detection.py  # Object detector
│   │   └── reasoning.py  # Reasoning layer (Part B)
│   └── training/
│       ├── train.py      # Training script
│       └── evaluate.py   # Evaluation script
├── models/
│   └── best.pt           # Trained model (64MB)
├── docs/
│   └── memo.md           # Technical memo
├── API_USAGE.md          # API documentation
├── requirements.txt      # Dependencies
└── Dockerfile            # Docker configuration
```

---

## 🌐 Deployment

### Live Deployment

**Deployed on Railway.app**: [Your Railway URL Here]
- **Status**: Active ✅
- **Region**: US West
- **Docs**: `/docs` endpoint for interactive API testing

### Deploy Your Own

**Railway.app** (Recommended - 8GB RAM free tier):
```bash
# 1. Push to GitHub
git push origin main

# 2. Connect at https://railway.app
# 3. Deploy from GitHub repo
# 4. Add environment variables:
#    - MODEL_PATH=models/best.pt
#    - CONFIDENCE_THRESHOLD=0.5
#    - PORT=8000
# 5. Railway auto-detects Dockerfile and deploys!
```

**Note**: Model auto-downloads from GitHub LFS on first startup (~2 minutes).

### Local Docker

```bash
docker build -t classroom-api .
docker run -p 7860:7860 classroom-api
```

---

## 📝 Submission Deliverables

- ✅ **Source Code**: Complete training + evaluation + API implementation
- ✅ **Model Weights**: `models/best.pt` (64MB, auto-downloaded via Git LFS)
- ✅ **Technical Memo**: `docs/memo.md` with 5 failure case analyses
- ✅ **API Documentation**: `API_USAGE.md` with sample requests/responses
- ✅ **Docker Support**: Production-ready `Dockerfile`
- ✅ **Live Deployment**: Railway.app with 8GB RAM
- ✅ **No Frameworks**: Pure Python reasoning layer (PDF requirement)

---

## 📞 Support

- **GitHub Repository**: [RAP-Submission](https://github.com/Venkat7123/RAP-Submission)
- **Issues**: [GitHub Issues](https://github.com/Venkat7123/RAP-Submission/issues)
- **Documentation**: See `API_USAGE.md` for detailed endpoint examples

---

## 📄 License

MIT License - See LICENSE file

---

**Author**: Venkatachalam S  
**Date**: September 13, 2026  
**Track**: Computer Vision + Applied ML Engineering
