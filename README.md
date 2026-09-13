# Classroom Object Detection System

**Project**: RT-DETR-based Object Detection with Reasoning API  
**Domain**: Classroom/Educational Space Detection  
**Classes**: board, chair, desk, fan

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

## Setup Instructions
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (manual step - see Dataset Guide)
# Place in data/raw/

# Run training
python src/training/train.py

# Start API server
uvicorn src.api.main:app --reload
```

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

## Deliverables Checklist
- [x] GitHub repository with all code
- [x] Trained model weights (downloadable)
- [x] 2-page technical memo
- [x] API usage instructions
- [x] Docker setup (bonus)
- [x] Reproducible training instructions
