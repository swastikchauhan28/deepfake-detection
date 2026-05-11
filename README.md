# Visual Deepfake Detection
### Swin Transformer + LSTM + Attention Mechanism
**B.Tech Major Project — Week 9 Progress Stage**

---

## Architecture

```
Input Video
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Frame Sampling  (16 frames, uniformly spaced)      │
│  Face Extraction (Haar Cascade face crop)           │
│  Preprocessing   (resize 224×224, ImageNet norm)    │
└─────────────────────────────────────────────────────┘
    │  (B, T=16, 3, 224, 224)
    ▼
┌─────────────────────────────────────────────────────┐
│  Swin Transformer (swin_tiny — ImageNet pretrained) │
│  Applied to each frame independently                │
│  Output: 768-d feature per frame                   │
└─────────────────────────────────────────────────────┘
    │  (B, T, 768)
    ▼
┌─────────────────────────────────────────────────────┐
│  Bidirectional LSTM  (2 layers, hidden=256)         │
│  Captures temporal change across frames             │
│  Output: 512-d per frame  (256×2 bidirectional)    │
└─────────────────────────────────────────────────────┘
    │  (B, T, 512)
    ▼
┌─────────────────────────────────────────────────────┐
│  Additive Attention                                 │
│  Learns which frames are most "suspicious"          │
│  Produces weighted context vector                   │
└─────────────────────────────────────────────────────┘
    │  (B, 512)
    ▼
┌─────────────────────────────────────────────────────┐
│  Classifier MLP  →  Linear(512→128) → Linear(128→2)│
│  Output: logits for [REAL, FAKE]                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
  Prediction: REAL / FAKE + confidence %
```

---

## Folder Structure

```
deepfake_detection/
├── data/
│   ├── train/
│   │   ├── real/         ← real .mp4 videos
│   │   └── fake/         ← manipulated .mp4 videos
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
├── checkpoints/          ← saved during training
│   ├── epoch_01.pth
│   ├── best_model.pth
│   └── training_log.csv
│
├── dataset.py            ← data loading, face extraction, transforms
├── model.py              ← Swin + LSTM + Attention architecture
├── train.py              ← training loop
├── inference.py          ← single-video prediction
├── evaluate.py           ← test-set metrics
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone / download the project
cd deepfake_detection

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place videos in data/train/real, data/train/fake, etc.
```

**Recommended dataset:** FaceForensics++ (c23 compression, ~1000 videos per class)
Download: https://github.com/ondyari/FaceForensics

---

## How to Run

### Train
```bash
python train.py \
    --train_dir  data/train \
    --val_dir    data/val \
    --epochs     10 \
    --batch_size 4 \
    --lr         1e-4
```

### Evaluate on test set
```bash
python evaluate.py \
    --test_dir   data/test \
    --checkpoint checkpoints/best_model.pth
```

### Predict on a single video
```bash
python inference.py \
    --video      path/to/video.mp4 \
    --checkpoint checkpoints/best_model.pth
```

### Verify model architecture (no data needed)
```bash
python model.py
```

---

## Expected Output

### Training
```
[Epoch 1/10] Train Loss: 0.6821  Train Acc: 58.3%  | Val Loss: 0.6534  Val Acc: 62.1%  (142s)
[Epoch 5/10] Train Loss: 0.4912  Train Acc: 76.8%  | Val Loss: 0.5103  Val Acc: 74.2%
[Epoch 10/10] Train Loss: 0.3841  Train Acc: 83.5%  | Val Loss: 0.4287  Val Acc: 79.6%
```

### Inference
```
=============================================
  Prediction   : FAKE
  REAL confidence : 18.3%
  FAKE confidence : 81.7%
  Most suspicious frame index : 9
=============================================

Attention weights across frames:
  Frame 01  0.0421  ████
  Frame 09  0.1283  ████████████
  Frame 14  0.0987  █████████
```

### Evaluation
```
=============================================
  Accuracy  : 79.40%
  Precision : 81.20%
  Recall    : 77.50%
  F1 Score  : 79.30%

  Confusion Matrix:
              Pred:REAL  Pred:FAKE
  Actual:REAL      412        88
  Actual:FAKE      114       386
=============================================
```

---

## Key Design Decisions (for Viva)

| Component | Choice | Why |
|---|---|---|
| Spatial encoder | Swin Transformer (tiny) | Shift-window attention; efficient for 224×224; ImageNet pretrained |
| Temporal encoder | Bidirectional LSTM | Captures sequential dependencies both forward & backward |
| Pooling | Additive Attention | Learns to focus on most manipulated frames instead of averaging |
| Face detection | Haar Cascade | Simple, no extra model weights, sufficient for college project |
| Optimiser | Adam + StepLR | Standard choice; LR decay every 4 epochs for stability |
| Frames | 16 per video | Balance between temporal coverage and memory/compute |

---


