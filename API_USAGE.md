# 🚀 API Usage Instructions

## Classroom Object Detection & Reasoning API

---

## 📦 Setup & Installation

### Prerequisites
- Python 3.10+
- 8GB RAM minimum
- GPU recommended (CPU works but slower)

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/classroom-detection-api.git
cd classroom-detection-api
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Model Exists
The trained model should be at: `models/best.pt` (64MB)

If missing, download from: [Add your Google Drive/Dropbox link]

### 4. Start API Server
```bash
# Method 1: Direct
python app.py

# Method 2: Uvicorn
cd src/api
uvicorn main:app --host 0.0.0.0 --port 8000

# Method 3: Docker
docker build -t classroom-api -f Dockerfile .
docker run -p 8000:8000 classroom-api
```

Server starts at: `http://localhost:8000`

---

## 🔌 API Endpoints

### Base URL
- **Local**: `http://localhost:8000`
- **Deployed**: `https://your-deployment-url.com`

### Interactive Docs
Visit `/docs` for interactive Swagger UI: `http://localhost:8000/docs`

---

## 📡 Endpoint 1: `/detect` (Part A - Object Detection)

**Method**: `POST`  
**Content-Type**: `multipart/form-data`

### Request
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@classroom_image.jpg"
```

### Sample Response
```json
{
  "success": true,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "num_detections": 12,
  "detections": [
    {
      "x1": 120.5,
      "y1": 340.2,
      "x2": 245.8,
      "y2": 580.3,
      "confidence": 0.89,
      "class_name": "chair",
      "class_id": 1
    },
    {
      "x1": 450.0,
      "y1": 100.0,
      "x2": 800.0,
      "y2": 500.0,
      "confidence": 0.92,
      "class_name": "board",
      "class_id": 0
    },
    {
      "x1": 300.0,
      "y1": 400.0,
      "x2": 550.0,
      "y2": 700.0,
      "confidence": 0.85,
      "class_name": "desk",
      "class_id": 2
    }
  ],
  "processing_time_ms": 245.3
}
```

### Python Example
```python
import requests

url = "http://localhost:8000/detect"
files = {"file": open("classroom.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Found {result['num_detections']} objects:")
for det in result['detections']:
    print(f"  - {det['class_name']}: {det['confidence']:.2f}")
```

---

## 💬 Endpoint 2: `/ask` (Part B - Reasoning Layer)

**Method**: `POST`  
**Content-Type**: `multipart/form-data`

### Request
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@classroom_image.jpg" \
  -F "question=How many chairs are in this image?"
```

### Sample Response (Counting)
```json
{
  "success": true,
  "question": "How many chairs are in this image?",
  "answer": "I can see 8 chairs in this image.",
  "confidence": "high",
  "used_detection": true,
  "num_objects_analyzed": 12,
  "processing_time_ms": 312.5
}
```

### Sample Response (Existence Check)
```json
{
  "success": true,
  "question": "Is there a whiteboard visible?",
  "answer": "Yes, there is 1 board visible.",
  "confidence": "high",
  "used_detection": true,
  "num_objects_analyzed": 12,
  "processing_time_ms": 298.1
}
```

### Sample Response (Comparison)
```json
{
  "success": true,
  "question": "Are there more desks or chairs?",
  "answer": "There are more chairs (8) than desks (4).",
  "confidence": "high",
  "used_detection": true,
  "num_objects_analyzed": 12,
  "processing_time_ms": 305.7
}
```

### Sample Response (Insufficient Information)
```json
{
  "success": true,
  "question": "Is anyone sitting at the desks?",
  "answer": "Insufficient information to answer this question confidently. I can detect chairs, desks, and boards in classroom images, but this question requires information I cannot extract from the detections.",
  "confidence": "insufficient",
  "used_detection": true,
  "num_objects_analyzed": 12,
  "processing_time_ms": 287.3
}
```

### Python Example
```python
import requests

url = "http://localhost:8000/ask"
files = {"file": open("classroom.jpg", "rb")}
data = {"question": "How many chairs are visible?"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Q: {result['question']}")
print(f"A: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

---

## 🧪 Example Questions

### Counting Questions
- "How many chairs are in this image?"
- "How many desks can you see?"
- "Count the number of fans"

### Existence Questions
- "Is there a whiteboard visible?"
- "Are there any fans in the room?"
- "Does this classroom have a board?"

### Comparison Questions
- "Are there more desks or chairs?"
- "What's the most common object?"
- "Which object appears least?"

### Insufficient Info (Guardrail Test)
- "Is anyone sitting at the desks?" ← No person detection
- "What color is the wall?" ← Not detectable
- "Is the room occupied?" ← Requires inference beyond detection

---

## 🏥 Health Check Endpoints

### `/` - Root
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "message": "Classroom Object Detection API",
  "version": "1.0.0",
  "endpoints": {
    "detect": "/detect (POST) - Object detection",
    "ask": "/ask (POST) - Natural language reasoning",
    "docs": "/docs - Interactive API documentation"
  },
  "status": "healthy"
}
```

### `/health` - Detailed Health
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_path": "models/best.pt",
  "model_loaded": true,
  "reasoner_loaded": true
}
```

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t classroom-detection-api -f Dockerfile .
```

### Run Container
```bash
docker run -d \
  -p 8000:8000 \
  --name classroom-api \
  classroom-detection-api
```

### Test
```bash
curl http://localhost:8000/health
```

---

## 🌐 Deployed URL (Submission)

**Live API Endpoint**: [YOUR DEPLOYED URL HERE]

Examples:
- Health: `https://your-url.com/health`
- Interactive Docs: `https://your-url.com/docs`
- Detection: POST to `https://your-url.com/detect`
- Reasoning: POST to `https://your-url.com/ask`

---

## ⚠️ Troubleshooting

### Model Not Found
```
FileNotFoundError: Model not found: models/best.pt
```
**Solution**: Download model weights and place in `models/` folder

### Port Already in Use
```
Error: [Errno 48] Address already in use
```
**Solution**: Change port in `.env` file or kill existing process

### GPU/CUDA Errors
**Solution**: API works on CPU. Set in code: `device='cpu'`

### Large Image Upload Fails
**Solution**: Resize images < 10MB before uploading

---

## 📞 Support

For issues or questions:
- GitHub Issues: [Your repo URL]
- Email: [Your email]

---

## 📄 License

MIT License - See LICENSE file

---

**Last Updated**: September 13, 2026  
**Author**: Venkatachalam S
