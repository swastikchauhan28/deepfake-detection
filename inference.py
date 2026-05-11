"""
inference.py — Run prediction on a single video
Week 9 stage: Simple CLI inference, prints result + confidence.

Usage:
    python inference.py --video path/to/video.mp4 \
                        --checkpoint checkpoints/best_model.pth
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np

from dataset import sample_frames, extract_face, get_transforms, NUM_FRAMES
from model   import DeepfakeDetector


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Deepfake Inference")
    p.add_argument("--video",      type=str, required=True,
                   help="Path to the video file to analyse")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth",
                   help="Path to saved model weights (.pth)")
    p.add_argument("--n_frames",   type=int, default=NUM_FRAMES)
    p.add_argument("--lstm_hidden",type=int, default=256)
    return p.parse_args()


# ──────────────────────────────────────────────
# Build input tensor from a video file
# ──────────────────────────────────────────────
def video_to_tensor(video_path: str, n_frames: int):
    """
    Reads a video, extracts n_frames face-cropped frames,
    and returns a tensor of shape (1, T, 3, 224, 224) ready for the model.
    """
    raw_frames = sample_frames(video_path, n_frames)

    if not raw_frames:
        raise ValueError(f"Could not read frames from: {video_path}")

    # Pad if video is shorter than n_frames
    while len(raw_frames) < n_frames:
        raw_frames.append(raw_frames[-1])

    transform = get_transforms("val")
    processed = []

    for frame in raw_frames[:n_frames]:
        import cv2
        face   = extract_face(frame)
        face   = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = transform(face)
        processed.append(tensor)

    clip = torch.stack(processed)         # (T, 3, 224, 224)
    return clip.unsqueeze(0)              # (1, T, 3, 224, 224)


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def predict(video_path: str, checkpoint: str,
            n_frames: int, lstm_hidden: int):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Video : {video_path}")
    print(f"Model : {checkpoint}\n")

    # ── Load model ────────────────────────────
    model = DeepfakeDetector(
        n_frames    = n_frames,
        lstm_hidden = lstm_hidden,
        pretrained  = False          # weights come from checkpoint
    ).to(device)

    state = torch.load(checkpoint, map_location=device)
    # Handle both full-checkpoint dicts and bare state dicts
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()

    # ── Prepare input ─────────────────────────
    clip = video_to_tensor(video_path, n_frames).to(device)

    # ── Forward pass ──────────────────────────
    with torch.no_grad():
        logits, alpha = model(clip)
        probs = F.softmax(logits, dim=1)[0]   # (2,)

    real_conf = probs[0].item() * 100
    fake_conf = probs[1].item() * 100
    pred_label = "FAKE" if fake_conf > real_conf else "REAL"

    # ── Most suspicious frame ─────────────────
    top_frame_idx = alpha[0].argmax().item()

    # ── Report ────────────────────────────────
    print("=" * 45)
    print(f"  Prediction   : {pred_label}")
    print(f"  REAL confidence : {real_conf:.1f}%")
    print(f"  FAKE confidence : {fake_conf:.1f}%")
    print(f"  Most suspicious frame index : {top_frame_idx}")
    print("=" * 45)

    # Attention weights per frame (useful for debugging/viva demo)
    attn_np = alpha[0].cpu().numpy()
    print("\nAttention weights across frames:")
    for i, w in enumerate(attn_np):
        bar = "█" * int(w * 100)
        print(f"  Frame {i+1:02d}  {w:.4f}  {bar}")

    return pred_label, fake_conf


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    predict(
        video_path  = args.video,
        checkpoint  = args.checkpoint,
        n_frames    = args.n_frames,
        lstm_hidden = args.lstm_hidden
    )
