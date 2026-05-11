
"""
app.py — DeepGuard | Visual Deepfake Detection
Clean minimal UI — Upload → Detect → Result
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time
import tempfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="DeepGuard",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg:     #09090b;
    --card:   #18181b;
    --border: #27272a;
    --brd2:   #3f3f46;
    --accent: #6366f1;
    --danger: #ef4444;
    --safe:   #22c55e;
    --text:   #fafafa;
    --muted:  #71717a;
    --dim:    #52525b;
    --mono:   'JetBrains Mono', monospace;
    --sans:   'Inter', sans-serif;
}
html,body,[class*="css"]{ font-family:var(--sans)!important; background:var(--bg)!important; color:var(--text)!important; }
.stApp{ background:var(--bg)!important; }
#MainMenu,footer,header{ visibility:hidden; }
.block-container{ max-width:660px!important; padding:3rem 1.5rem 5rem!important; margin:0 auto!important; }
[data-testid="stSidebar"]{ display:none!important; }
[data-testid="stFileUploader"]{ background:var(--card)!important; border:1.5px dashed var(--brd2)!important; border-radius:16px!important; }
[data-testid="stFileUploader"]:hover{ border-color:var(--accent)!important; }
[data-testid="stFileUploader"] label{ display:none!important; }
.stButton>button{ background:var(--accent)!important; color:#fff!important; border:none!important; border-radius:12px!important; font-family:var(--sans)!important; font-size:0.9rem!important; font-weight:600!important; padding:0.7rem 2rem!important; width:100%!important; transition:all 0.15s!important; box-shadow:0 1px 2px rgba(0,0,0,0.4)!important; }
.stButton>button:hover{ background:#4f46e5!important; transform:translateY(-1px)!important; box-shadow:0 4px 16px rgba(99,102,241,0.35)!important; }
.stButton>button:disabled{ background:var(--border)!important; color:var(--dim)!important; transform:none!important; box-shadow:none!important; }
.stSpinner>div{ border-top-color:var(--accent)!important; }
::-webkit-scrollbar{ width:4px; }
::-webkit-scrollbar-thumb{ background:var(--brd2); border-radius:2px; }
details summary{ font-size:0.8rem; color:var(--muted); cursor:pointer; }
@keyframes fadeUp{ from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
@keyframes blink{ 0%,100%{opacity:1} 50%{opacity:0.3} }
.fu{ animation:fadeUp 0.45s ease forwards; }
</style>
""", unsafe_allow_html=True)

# ── Model imports ──────────────────────────────────────────────────
MODEL_AVAILABLE = False
try:
    from model import DeepfakeDetector
    from dataset import sample_frames, extract_face, get_transforms
    MODEL_AVAILABLE = True
except ImportError:
    pass

@st.cache_resource
def load_model(cp, nf, lh):
    if not MODEL_AVAILABLE: return None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DeepfakeDetector(n_frames=nf, lstm_hidden=lh, pretrained=False).to(dev)
    if os.path.exists(cp):
        s = torch.load(cp, map_location=dev)
        m.load_state_dict(s["model_state_dict"] if "model_state_dict" in s else s)
        m.eval(); return m
    return None

def video_to_tensor(path, nf):
    frames = sample_frames(path, nf)
    if not frames: return None, []
    while len(frames) < nf: frames.append(frames[-1])
    tf = get_transforms("val")
    proc, imgs = [], []
    for f in frames[:nf]:
        face = extract_face(f)
        rgb  = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        imgs.append(rgb); proc.append(tf(rgb))
    return torch.stack(proc).unsqueeze(0), imgs

def run_inference(model, path, nf):
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, imgs = video_to_tensor(path, nf)
    if clip is None: return None
    clip = clip.to(dev)
    with torch.no_grad():
        logits, alpha = model(clip)
        probs = F.softmax(logits, dim=1)[0]
    return {"real_conf": probs[0].item()*100, "fake_conf": probs[1].item()*100,
            "prediction": "FAKE" if probs[1]>probs[0] else "REAL",
            "alpha": alpha[0].cpu().numpy(), "face_images": imgs,
            "top_frame": alpha[0].argmax().item()}

def demo_inference(fname=""):
    time.sleep(1.8)
    is_fake  = "fake" in fname.lower()
    fake_conf = np.random.uniform(72,94) if is_fake else np.random.uniform(5,27)
    alpha = np.random.dirichlet(np.ones(8)*0.6)
    return {"real_conf": 100-fake_conf, "fake_conf": fake_conf,
            "prediction": "FAKE" if fake_conf>50 else "REAL",
            "alpha": alpha, "face_images": [], "top_frame": int(np.argmax(alpha))}

def plot_attention(weights):
    n = len(weights); mi = int(np.argmax(weights))
    fig, ax = plt.subplots(figsize=(7, 2.1))
    fig.patch.set_facecolor("#18181b"); ax.set_facecolor("#18181b")
    cols = ["#ef4444" if i==mi else "#6366f1" for i in range(n)]
    ax.bar(range(n), weights, color=cols, width=0.5, zorder=3)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"F{i+1}" for i in range(n)], color="#52525b", fontsize=8, fontfamily="monospace")
    ax.tick_params(axis="y", colors="#52525b", labelsize=7)
    ax.spines[:].set_color("#27272a")
    ax.grid(axis="y", color="#27272a", linewidth=0.5, zorder=0)
    for i,(x,h) in enumerate(zip(range(n), weights)):
        ax.text(x, h+0.002, f"{h:.3f}", ha="center", va="bottom",
                color="#ef4444" if i==mi else "#3f3f46", fontsize=5.5, fontfamily="monospace")
    plt.tight_layout(pad=0.3); return fig

# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:2rem 0 2.8rem;">
  <div style="display:inline-flex;align-items:center;gap:8px;
              background:#18181b;border:1px solid #27272a;
              border-radius:100px;padding:5px 16px;margin-bottom:2rem;">
    <span style="width:7px;height:7px;border-radius:50%;background:#22c55e;
                 display:inline-block;animation:blink 2s infinite;"></span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;
                 color:#52525b;letter-spacing:1.5px;">DEEPFAKE DETECTOR</span>
  </div>
  <div style="font-size:2.75rem;font-weight:800;letter-spacing:-1.5px;
              line-height:1.1;margin-bottom:1rem;">
    <span style="background:linear-gradient(135deg,#fafafa 40%,#52525b);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      Is this video real?
    </span>
  </div>
  <p style="font-size:0.9rem;color:#71717a;max-width:400px;
            margin:0 auto;line-height:1.75;font-weight:400;">
    Upload a face video. Our AI detects manipulation in seconds.
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════════
with st.expander("⚙️  Settings", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1: n_frames     = st.select_slider("Frames", [4,8,12,16], value=8)
    with c2: lstm_hidden  = st.selectbox("LSTM", [128,256,512], index=0)
    with c3: checkpoint_path = st.text_input("Checkpoint", value="checkpoints/best_model.pth")
    use_demo = st.toggle("Demo Mode (no model required)", value=not MODEL_AVAILABLE)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  UPLOAD
# ══════════════════════════════════════════════════════════════════
uploaded = st.file_uploader("upload", type=["mp4","avi","mov"], label_visibility="collapsed")

if not uploaded:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.5rem;pointer-events:none;">
      <div style="font-size:2rem;opacity:0.18;margin-bottom:0.5rem;">📹</div>
      <div style="font-size:0.82rem;color:#3f3f46;font-weight:500;">
        Drop video here or click to browse
      </div>
      <div style="font-size:0.7rem;color:#27272a;margin-top:4px;">MP4 · AVI · MOV</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  FILE INFO + BUTTON
# ══════════════════════════════════════════════════════════════════
if uploaded:
    st.markdown(f"""
    <div class="fu" style="display:flex;align-items:center;justify-content:space-between;
         background:#18181b;border:1px solid #27272a;border-radius:12px;
         padding:0.75rem 1rem;margin:0.8rem 0 1rem;">
      <div style="display:flex;align-items:center;gap:0.75rem;">
        <div style="width:36px;height:36px;background:#27272a;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;font-size:1rem;">🎬</div>
        <div>
          <div style="font-size:0.82rem;font-weight:500;color:#fafafa;
                      max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
            {uploaded.name}
          </div>
          <div style="font-size:0.68rem;color:#52525b;margin-top:1px;">
            {uploaded.size/1024:.1f} KB &nbsp;·&nbsp; {'Demo' if use_demo else ('GPU' if torch.cuda.is_available() else 'CPU')}
          </div>
        </div>
      </div>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#22c55e;
                   background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.18);
                   padding:3px 10px;border-radius:100px;">Ready</span>
    </div>
    """, unsafe_allow_html=True)

    btn = st.button("Analyse Video", use_container_width=True)

    if btn:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read()); tmp_path = tmp.name

        with st.spinner("Analysing frames..."):
            if use_demo:
                result = demo_inference(uploaded.name)
            else:
                m = load_model(checkpoint_path, n_frames, lstm_hidden)
                result = run_inference(m, tmp_path, n_frames) if m else None

        os.unlink(tmp_path)

        if result is None:
            st.error("Could not process. Enable Demo Mode or check model path.")
        else:
            st.session_state["result"] = result

# ══════════════════════════════════════════════════════════════════
#  RESULT
# ══════════════════════════════════════════════════════════════════
if "result" in st.session_state and uploaded:
    r       = st.session_state["result"]
    is_fake = r["prediction"] == "FAKE"
    fc      = r["fake_conf"]
    rc      = r["real_conf"]

    col     = "#ef4444" if is_fake else "#22c55e"
    bg      = "rgba(239,68,68,0.06)"  if is_fake else "rgba(34,197,94,0.06)"
    ring    = "rgba(239,68,68,0.18)"  if is_fake else "rgba(34,197,94,0.18)"
    label   = "Deepfake"              if is_fake else "Authentic"
    desc    = "This video appears to be AI-manipulated" \
              if is_fake else "No deepfake manipulation detected"
    icon    = "⚠" if is_fake else "✓"

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#27272a,transparent);margin-bottom:1.8rem;"></div>', unsafe_allow_html=True)

    # Verdict card
    st.markdown(f"""
    <div class="fu" style="background:{bg};border:1.5px solid {ring};
         border-radius:20px;padding:2.5rem 1.5rem;text-align:center;margin-bottom:1.2rem;">
      <div style="width:52px;height:52px;border-radius:50%;background:{col};color:#fff;
                  font-size:1.4rem;font-weight:700;display:flex;align-items:center;
                  justify-content:center;margin:0 auto 1.2rem;
                  box-shadow:0 0 0 8px {ring};">
        {icon}
      </div>
      <div style="font-size:2rem;font-weight:800;letter-spacing:-0.5px;
                  color:{col};margin-bottom:0.35rem;">{label}</div>
      <div style="font-size:0.82rem;color:#71717a;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence
    st.markdown(f"""
    <div class="fu" style="background:#18181b;border:1px solid #27272a;
         border-radius:16px;padding:1.4rem 1.5rem;margin-bottom:1.2rem;">
      <div style="font-size:0.62rem;font-family:'JetBrains Mono',monospace;
                  color:#3f3f46;letter-spacing:2px;text-transform:uppercase;
                  margin-bottom:1.2rem;">Confidence Score</div>
      <div style="margin-bottom:1rem;">
        <div style="display:flex;justify-content:space-between;
                    font-size:0.8rem;margin-bottom:6px;">
          <span style="color:#ef4444;font-weight:500;">Fake</span>
          <span style="color:#ef4444;font-family:'JetBrains Mono',monospace;
                       font-weight:700;">{fc:.1f}%</span>
        </div>
        <div style="height:6px;background:#27272a;border-radius:3px;overflow:hidden;">
          <div style="height:100%;width:{fc:.1f}%;border-radius:3px;
                      background:linear-gradient(90deg,#ef4444,#f87171);"></div>
        </div>
      </div>
      <div>
        <div style="display:flex;justify-content:space-between;
                    font-size:0.8rem;margin-bottom:6px;">
          <span style="color:#22c55e;font-weight:500;">Real</span>
          <span style="color:#22c55e;font-family:'JetBrains Mono',monospace;
                       font-weight:700;">{rc:.1f}%</span>
        </div>
        <div style="height:6px;background:#27272a;border-radius:3px;overflow:hidden;">
          <div style="height:100%;width:{rc:.1f}%;border-radius:3px;
                      background:linear-gradient(90deg,#22c55e,#4ade80);"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Attention chart
    st.markdown(f"""
    <div class="fu" style="background:#18181b;border:1px solid #27272a;
         border-radius:16px;padding:1.4rem 1.5rem 0.6rem;">
      <div style="font-size:0.62rem;font-family:'JetBrains Mono',monospace;
                  color:#3f3f46;letter-spacing:2px;text-transform:uppercase;
                  margin-bottom:2px;">Frame Attention</div>
      <div style="font-size:0.72rem;color:#3f3f46;margin-bottom:0.8rem;">
        Most suspicious frame:
        <span style="color:#ef4444;font-family:'JetBrains Mono',monospace;
                     font-weight:600;">#{r['top_frame']+1}</span>
      </div>
    """, unsafe_allow_html=True)

    fig = plot_attention(r["alpha"])
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown("</div>", unsafe_allow_html=True)

    # Again button
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("Try Another Video", use_container_width=True):
        del st.session_state["result"]
        st.rerun()

# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:3rem 0 1rem;margin-top:3rem;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
               color:#27272a;letter-spacing:1px;">
    DeepGuard · Swin Transformer + BiLSTM + Attention
  </span>
</div>
""", unsafe_allow_html=True)