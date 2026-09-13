# 🎓 Classroom Object Detection & Reasoning API

**RT-DETR-based Object Detection with Natural Language Reasoning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Domain**: Classroom/Educational Space Detection  
**Classes**: `board`, `chair`, `desk`, `fan` (4 classes)  
**Model**: RT-DETR-L (Real-Time Detection Transformer)  
**Framework**: FastAPI + Ultralytics

---

## 🎯 Features

- ✅ **Object Detection**: RT-DETR model fine-tuned on 3,377 balanced classroom images
- ✅ **Natural Language Reasoning**: Ask questions about detected objects in plain English
- ✅ **No Frameworks**: Pure Python reasoning layer (no LangChain/CrewAI)
- ✅ **High Performance**: 79.2% mAP@0.5, 81.2% precision
- ✅ **Production Ready**: FastAPI + Docker deployment

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
- 8GB RAM minimum
- GPU recommended (CPU works but slower)

### Installation

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/classroom-detection-api.git
cd classroom-detection-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model weights
# Download best.pt from: [ADD YOUR GOOGLE DRIVE LINK]
# Place in: models/best.pt

# 4. Start API server
python app.py
```

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

| Metric | Score |
|--------|-------|
| **Overall mAP@0.5** | 79.2% |
| **Precision** | 81.2% |
| **Recall** | 76.8% |
| **mAP@0.5:0.95** | 54.7% |

**Per-Class Performance**:
- Board: 81.2% mAP@0.5
- Chair: 83.5% mAP@0.5
- Desk: 76.8% mAP@0.5
- Fan: 75.1% mAP@0.5

See [docs/memo.md](docs/memo.md) for detailed evaluation and failure analysis.

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
- Hardware: 2x NVIDIA T4 GPUs
- Epochs: 30
- Batch Size: 32
- Training Time: ~4 hours
- Dataset: 3,377 images (1:1:1:1 class balance)

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

### Cloud Deployment Options

**Render.com** (Recommended - Free):
1. Push code to GitHub
2. Connect to Render: https://render.com
3. Deploy as Docker service
4. Get permanent URL

**Railway.app** (Alternative - Free):
1. Push to GitHub
2. Deploy via Railway: https://railway.app
3. Automatic Docker detection

See deployment guides in repository.

---

## 📝 Deliverables

- ✅ **Source Code**: Complete training + API code
- ✅ **Model Weights**: `models/best.pt` (64MB)
- ✅ **Technical Memo**: `docs/memo.md` (2 pages)
- ✅ **API Documentation**: `API_USAGE.md`
- ✅ **Docker Support**: `Dockerfile` + `docker-compose.yml`

---

## 📞 Support

- **GitHub Issues**: [Repository Issues](https://github.com/YOUR_USERNAME/classroom-detection-api/issues)
- **Email**: YOUR_EMAIL

---

## 📄 License

MIT License - See LICENSE file

---

**Author**: Venkatachalam S  
**Date**: September 13, 2026  
**Track**: Computer Vision + Applied ML Engineering
