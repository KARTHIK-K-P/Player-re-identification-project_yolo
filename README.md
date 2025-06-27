# 🏈 Player Re-Identification in Sports Footage


A state-of-the-art computer vision system for identifying and tracking players across different camera angles in sports footage using YOLO detection and deep learning-based re-identification.

## 🌟 Features

- ⚡ **Real-time Player Detection** using YOLOv8/YOLOv5
- 🔍 **Cross-Camera Re-identification** with deep feature matching
- 📹 **Multi-object Tracking** with temporal consistency
- 🎯 **High Accuracy** player matching across different viewpoints
- 🖥️ **GPU Acceleration** with CUDA support
- 📊 **Comprehensive Logging** and result visualization
- ⚙️ **Flexible Configuration** via YAML files

## 📁 Project Structure

```
PLAYER_REID/
├── 📄 README.md                # This file
├── 📁 src/                    # Source code
│   ├── 🐍 main.py            # Main execution script
│   ├── 🐍 player_detector.py # YOLO-based detection
│   ├── 🐍 reid_matcher.py    # Re-identification matching
│   ├── 🐍 tracker.py         # Player tracking
│   ├── 🐍 visualizer.py      # Output visualization
│   ├── 🐍 feature_extractor.py # Feature extraction
│   └── ⚙️ config.yaml        # Configuration file
├── 📁 data/                  # Input data
│   ├── 📁 models/           # YOLO model weights
│   └── 📁 videos/           # Input videos
├── 📁 output/                # Generated results
│   ├── 📁 detections/       # Detection JSON files
│   ├── 📁 features/         # Feature vectors
│   ├── 📁 videos/           # Output videos
│   └── 📄 matched_pairs.json
└── 📁 logs/                  # Execution logs
```

## 🚀 Quick Start

### 1️⃣ Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (recommended)
- **4GB+ RAM**
- **2GB+ disk space**

### 2️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/player-reid-sports.git
cd player-reid-sports

# Install dependencies
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy scikit-learn pyyaml matplotlib pillow

# Or install from requirements.txt
pip install -r requirements.txt
```

### 3️⃣ Run the System

```bash
# Basic execution (uses config.yaml defaults)
python src/main.py

# Process your own video
python src/main.py --input data/videos/your_video.mp4

# With GPU acceleration
python src/main.py --device cuda --confidence 0.6
```

## ⚙️ Configuration

The system uses an integrated YAML configuration file (`src/config.yaml`):

<details>
<summary>📋 <strong>View Full Configuration Options</strong></summary>

```yaml
# Player Re-identification Configuration
project_name: "Player_ReID_Sports"
version: "1.0"

# Input/Output Settings
paths:
  input_videos: "data/videos/"
  output_dir: "output/"
  models_dir: "data/models/"
  logs_dir: "logs/"

# Video Processing
video:
  input_file: "broadcast.mp4"
  output_file: "output_reid.mp4"
  frame_skip: 1
  resize_width: 1280
  resize_height: 720
  fps: 30

# YOLO Detection
detection:
  model_name: "yolov8n.pt"
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: "cuda"
  imgsz: 640

# Feature Extraction
feature_extraction:
  model_type: "resnet50"
  feature_dim: 2048
  normalize_features: true
  batch_size: 32

# Re-identification
reid:
  similarity_metric: "cosine"
  similarity_threshold: 0.7
  temporal_window: 30

# Tracking
tracking:
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

# Visualization
visualization:
  show_bboxes: true
  show_ids: true
  bbox_thickness: 2
  colors:
    bbox: [0, 255, 0]
    text: [255, 255, 255]
```

</details>

## 🎯 Usage Examples

### Basic Usage
```bash
# Default processing
python src/main.py

# Process specific video
python src/main.py --input data/videos/football_match.mp4
```

### Advanced Usage
```bash
# High accuracy mode
python src/main.py --model yolov8s.pt --confidence 0.7 --similarity_threshold 0.8

# Fast processing mode
python src/main.py --model yolov8n.pt --confidence 0.4 --imgsz 416

# CPU-only processing
python src/main.py --device cpu

# Debug mode with visualization
python src/main.py --visualize --save_detections --verbose
```

### Batch Processing
```bash
# Process multiple videos
for video in data/videos/*.mp4; do
    python src/main.py --input "$video" --output "output/$(basename "$video" .mp4)_reid.mp4"
done
```

## 📊 Command Line Arguments

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--input` | str | Input video path | config.yaml |
| `--output` | str | Output video path | config.yaml |
| `--model` | str | YOLO model (yolov8n.pt, yolov8s.pt) | yolov8n.pt |
| `--confidence` | float | Detection confidence (0.0-1.0) | 0.5 |
| `--device` | str | Device (cuda, cpu, mps) | cuda |
| `--similarity_threshold` | float | Re-ID threshold (0.0-1.0) | 0.7 |
| `--visualize` | flag | Show real-time visualization | False |
| `--save_detections` | flag | Save detection results | False |
| `--verbose` | flag | Enable verbose logging | False |

## 📈 Output Files

| File | Location | Description |
|------|----------|-------------|
| **Processed Video** | `output/videos/` | Video with bounding boxes and IDs |
| **Detections** | `output/detections.json` | Player detection coordinates |
| **Re-ID Results** | `output/matched_pairs.json` | Matched player pairs |
| **Features** | `output/features.pkl` | Extracted feature vectors |
| **Logs** | `logs/` | Execution logs with timestamps |

### Sample Output Format

```json
{
  "detections": {
    "frame_001": [
      {
        "player_id": 1,
        "bbox": [100, 150, 200, 300],
        "confidence": 0.85,
        "features": [0.1, 0.2, ...]
      }
    ]
  },
  "matches": [
    {
      "player_id": 1,
      "frame_1": "001",
      "frame_2": "045",
      "similarity": 0.82
    }
  ]
}
```


## ⚡ Performance Optimization

### 🎯 For Better Accuracy:
- Use larger models: `yolov8s.pt`, `yolov8m.pt`
- Increase confidence: `--confidence 0.6`
- Higher similarity threshold: `--similarity_threshold 0.8`
- Larger input size: `--imgsz 640`

### 🚀 For Faster Processing:
- Use smaller model: `yolov8n.pt`
- Lower confidence: `--confidence 0.4`
- Smaller input size: `--imgsz 416`
- Skip frames: Set `frame_skip: 2` in config

### 💾 For Memory Efficiency:
- CPU processing: `--device cpu`
- Reduce batch size in config
- Lower image resolution

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04
- **CPU**: Intel i5 / AMD Ryzen 5
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Python**: 3.8+

### Recommended Requirements
- **OS**: Ubuntu 20.04 / Windows 11
- **CPU**: Intel i7 / AMD Ryzen 7
- **GPU**: NVIDIA GTX 1060 / RTX 3060
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **CUDA**: 11.8+

## 📜 Dependencies

```txt
torch>=1.12.0
torchvision>=0.13.0
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
PyYAML>=6.0
matplotlib>=3.5.0
Pillow>=9.0.0
`
