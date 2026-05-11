"""
model.py — Swin Transformer + LSTM + Attention Mechanism
Week 9 stage: Full architecture working, not yet fine-tuned

Architecture overview:
  Input: (B, T, 3, 224, 224)   — batch of T-frame video clips

  Step 1 — Swin Transformer (per-frame spatial encoder)
      Pre-trained Swin-Tiny backbone.
      Each frame → 768-d feature vector.
      Output: (B, T, 768)

  Step 2 — Bidirectional LSTM (temporal encoder)
      Captures how features CHANGE across frames.
      Output: (B, T, 2*hidden_size)  [bidirectional]

  Step 3 — Additive Attention (feature weighting)
      Learns which frames are most "suspicious".
      Output: (B, 2*hidden_size)  [weighted sum over T]

  Step 4 — Classifier MLP
      Two FC layers → binary logits (real vs fake)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_t, Swin_T_Weights


# ──────────────────────────────────────────────
# 1. Swin Transformer feature extractor
#    (one shared backbone applied per-frame)
# ──────────────────────────────────────────────
class SwinFrameEncoder(nn.Module):
    """
    Wraps the pre-trained Swin-Tiny model.
    Removes the classification head so we get raw 768-d embeddings.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = swin_t(weights=weights)

        # The Swin-Tiny head is backbone.head (a Linear layer).
        # We replace it with Identity to get the 768-d penultimate feature.
        self.feature_dim = backbone.head.in_features   # 768
        backbone.head    = nn.Identity()
        self.backbone    = backbone

    def forward(self, x):
        """
        x : (B*T, 3, 224, 224)
        Returns: (B*T, 768)
        """
        return self.backbone(x)


# ──────────────────────────────────────────────
# 2. Additive (Bahdanau-style) Attention
# ──────────────────────────────────────────────
class AdditiveAttention(nn.Module):
    """
    Computes a soft-attention weight over T time steps.

    Given LSTM output h of shape (B, T, H):
      score_t = tanh(W · h_t + b)
      alpha_t = softmax(v · score_t)
      context = Σ alpha_t * h_t

    This tells the model: "focus on the frames that are most informative".
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        """
        h      : (B, T, H)
        Returns: context (B, H), alpha (B, T)  ← alpha useful for visualisation
        """
        score = torch.tanh(self.W(h))       # (B, T, H)
        alpha = self.v(score).squeeze(-1)   # (B, T)
        alpha = F.softmax(alpha, dim=1)     # (B, T)  — sum to 1 over frames

        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, H)
        return context, alpha


# ──────────────────────────────────────────────
# 3. Full model
# ──────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    """
    End-to-end Swin + LSTM + Attention classifier.

    Args:
        n_frames      : number of frames per clip (T)
        lstm_hidden   : LSTM hidden units per direction
        lstm_layers   : number of stacked LSTM layers
        dropout       : dropout rate applied before classifier
        pretrained    : whether to use ImageNet pre-trained Swin weights
    """

    def __init__(self,
                 n_frames:    int  = 16,
                 lstm_hidden: int  = 256,
                 lstm_layers: int  = 2,
                 dropout:     float = 0.3,
                 pretrained:  bool  = True):
        super().__init__()

        self.n_frames = n_frames

        # ── Swin encoder ──────────────────────────────
        self.swin     = SwinFrameEncoder(pretrained=pretrained)
        feat_dim      = self.swin.feature_dim   # 768

        # ── Bidirectional LSTM ────────────────────────
        self.lstm = nn.LSTM(
            input_size  = feat_dim,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if lstm_layers > 1 else 0.0
        )
        lstm_out_dim = lstm_hidden * 2   # bidirectional doubles the output

        # ── Attention ─────────────────────────────────
        self.attention = AdditiveAttention(lstm_out_dim)

        # ── Classifier head ───────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)    # 2 classes: real (0) / fake (1)
        )

    def forward(self, x):
        """
        x : (B, T, 3, 224, 224)

        Returns:
            logits  : (B, 2)
            alpha   : (B, T)  — attention weights per frame
        """
        B, T, C, H, W = x.shape

        # ── 1. Extract per-frame Swin features ────────
        # Flatten batch + time so Swin sees (B*T, 3, H, W)
        x_flat   = x.view(B * T, C, H, W)
        feats    = self.swin(x_flat)            # (B*T, 768)
        feats    = feats.view(B, T, -1)         # (B, T, 768)

        # ── 2. LSTM over frame sequence ───────────────
        lstm_out, _ = self.lstm(feats)          # (B, T, 2*hidden)

        # ── 3. Attention pooling ──────────────────────
        context, alpha = self.attention(lstm_out)   # (B, 2*hidden), (B, T)

        # ── 4. Classify ───────────────────────────────
        logits = self.classifier(context)       # (B, 2)

        return logits, alpha


# ──────────────────────────────────────────────
# Quick sanity check (run: python model.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    model  = DeepfakeDetector(n_frames=16, lstm_hidden=256)
    dummy  = torch.randn(2, 16, 3, 224, 224)   # batch=2, 16 frames
    logits, alpha = model(dummy)
    print("Logits shape :", logits.shape)   # → (2, 2)
    print("Alpha  shape :", alpha.shape)    # → (2, 16)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total:,}")
