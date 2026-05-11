"""
dataset.py — Face extraction, frame sampling, and dataset loading
Week 9 stage: Basic working pipeline, no heavy augmentation yet
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
FRAME_SIZE   = 224          # Swin Transformer expects 224×224
NUM_FRAMES   = 16           # frames sampled per video clip
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ──────────────────────────────────────────────
# Helper: extract face region from a single frame
# ──────────────────────────────────────────────
def extract_face(frame: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """
    Detects the largest face in a BGR frame and returns a resized crop.
    Falls back to centre-crop if no face is detected.
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces  = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # Fallback: just resize the whole frame
        return cv2.resize(frame, (size, size))

    # Pick the largest detected face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face = frame[y:y + h, x:x + w]
    return cv2.resize(face, (size, size))


# ──────────────────────────────────────────────
# Helper: uniformly sample N frames from a video
# ──────────────────────────────────────────────
def sample_frames(video_path: str, n_frames: int = NUM_FRAMES) -> list:
    """
    Opens a video file and returns n_frames uniformly-spaced BGR frames.
    Returns an empty list if the video cannot be read.
    """
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


# ──────────────────────────────────────────────
# Preprocessing transforms (train vs. val/test)
# ──────────────────────────────────────────────
def get_transforms(split: str = "train"):
    """
    Returns a torchvision transform pipeline.
    Train: mild augmentation | Val/Test: clean normalisation only
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats — Swin was pre-trained on it
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ──────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────
class DeepfakeVideoDataset(Dataset):
    """
    Expects a root folder with two sub-folders:
        root/
          real/   ← real video files (.mp4 / .avi)
          fake/   ← manipulated video files

    Returns:
        frames_tensor : (NUM_FRAMES, 3, 224, 224)  — float32
        label         : 0 = real, 1 = fake          — long
    """

    def __init__(self, root_dir: str, split: str = "train",
                 n_frames: int = NUM_FRAMES):
        self.root_dir  = root_dir
        self.n_frames  = n_frames
        self.transform = get_transforms(split)
        self.samples   = []   # list of (video_path, label)

        for label, class_name in enumerate(["real", "fake"]):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".mp4", ".avi", ".mov")):
                    self.samples.append(
                        (os.path.join(class_dir, fname), label)
                    )

        print(f"[Dataset] {split} — {len(self.samples)} videos found in '{root_dir}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        raw_frames = sample_frames(video_path, self.n_frames)

        # If video is too short, pad with copies of the last frame
        while len(raw_frames) < self.n_frames:
            raw_frames.append(raw_frames[-1] if raw_frames else
                              np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8))

        # Face-crop every frame, then apply transforms
        processed = []
        for frame in raw_frames[:self.n_frames]:
            face = extract_face(frame, FRAME_SIZE)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)   # BGR → RGB
            tensor = self.transform(face)                   # (3, H, W)
            processed.append(tensor)

        frames_tensor = torch.stack(processed)              # (T, 3, H, W)
        return frames_tensor, torch.tensor(label, dtype=torch.long)


# ──────────────────────────────────────────────
# Convenience builder
# ──────────────────────────────────────────────
def build_dataloaders(train_dir: str, val_dir: str,
                      batch_size: int = 4, num_workers: int = 2):
    """
    Returns train_loader and val_loader ready for the training loop.
    batch_size=4 works comfortably on a 6 GB GPU; reduce to 2 for CPU.
    """
    train_ds = DeepfakeVideoDataset(train_dir, split="train")
    val_ds   = DeepfakeVideoDataset(val_dir,   split="val")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader
