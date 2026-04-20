"""
CycleGAN ukiyoe <-> photo
Run: streamlit run app.py
"""

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

APP_DIR = Path(__file__).parent.resolve()
CHECKPOINTS_DIR = APP_DIR / "checkpoints"
EXAMPLES_DIR = APP_DIR / "examples"

sys.path.insert(0, str(APP_DIR))
from models import CycleGAN

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_ckpt(path: Path):
    # epoch_{epoch}_{idf}_{cyc}_{gan}.pt
    parts = path.stem.split("_")
    try:
        epoch = int(parts[1])
        idf, cyc, gan = (int(p) for p in parts[2:5])
    except (ValueError, IndexError):
        return None
    return {"epoch": epoch, "idf": idf, "cyc": cyc, "gan": gan, "path": str(path)}


def list_checkpoints():
    items = [parse_ckpt(p) for p in sorted(CHECKPOINTS_DIR.glob("epoch_*.pt"))]
    return [it for it in items if it is not None]


def ckpt_label(c):
    return f"epoch {c['epoch']} · idf={c['idf']} · cyc={c['cyc']} · gan={c['gan']}"

st.set_page_config(
    page_title="Photo <-> Ukiyoe | CycleGAN",
    page_icon="🎨",
    layout="centered",
)

# ---------- background: pattern.monster japanese-pattern-1 (cool blue recolor) ----------
svg = (
    "<svg id='patternId' width='2000' height='2000' viewBox='0 0 2000 2000' xmlns='http://www.w3.org/2000/svg'>"
    "<defs><pattern id='a' patternUnits='userSpaceOnUse' width='69.283' height='40' patternTransform='scale(3) rotate(75)'>"
    "<rect x='0' y='0' width='100%' height='100%' fill='#eaf2f9'/>"
    "<path d=\"M46.189-20L57.736 0M46.189 20l11.547 20m-46.189 0l11.547 20M11.547 0l11.547 20m40.415 30H40.415M28.868 30H5.774m23.094-40H5.774m57.735 20H40.415m0 20L28.868 50m11.547-60L28.868 10m46.188 0L63.509 30M5.774 10L-5.773 30m75.056 10H46.189L34.64 20 46.19 0h23.094C73.13 6.667 76.98 13.333 80.83 20zM57.736 60H34.64L23.094 40l11.547-20h23.095c3.848 6.667 7.698 13.333 11.547 20L57.736 60zm0-40H34.64L23.094 0l11.547-20h23.095L69.283 0c-3.87 6.7-8.118 14.06-11.547 20zM34.64 60H11.547L0 40l11.547-20h23.094L46.19 40 34.64 60zm0-40H11.547L0 0l11.547-20h23.094L46.19 0 34.64 20zM23.094 40H0l-5.773-10-5.774-10L0 0h23.094l11.547 20-11.547 20z\" stroke-width='1' stroke='#5e90b8' fill='none'/>"
    "</pattern></defs>"
    "<rect width='800%' height='800%' transform='translate(-111,-183)' fill='url(#a)'/>"
    "</svg>"
)
bg_url = f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"

st.markdown(
    f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500&family=Inter:wght@300;400;500&display=swap');

  .stApp {{
    background-color: #eaf2f9;
    background-image: url("{bg_url}");
    background-repeat: no-repeat;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
  }}
  .block-container {{
    max-width: 880px;
    background: rgba(250,253,255,0.94);
    border-radius: 20px;
    padding: 2.4rem 3rem 3.5rem;
    backdrop-filter: blur(6px);
    box-shadow: 0 8px 40px rgba(30,58,95,0.10);
  }}
  [data-testid="collapsedControl"],
  section[data-testid="stSidebar"] {{ display: none; }}
  h1 {{
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 2.6rem !important;
    font-weight: 500 !important;
    color: #1a2838 !important;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    border-bottom: 1px solid #d0dce8;
    background: transparent;
  }}
  .stTabs [data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: .04em;
    text-transform: uppercase;
    color: #7b93a8;
    padding: 0.55rem 1.4rem;
    border-radius: 0;
    background: transparent;
    border-bottom: 2px solid transparent;
  }}
  .stTabs [aria-selected="true"] {{
    color: #1e3a5f !important;
    border-bottom-color: #1e3a5f !important;
    background: transparent !important;
  }}
  .stImage img {{
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(30,58,95,0.14);
  }}
  .pill {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border-radius: 999px;
    padding: 5px 15px;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    margin: 0.5rem 0 0.9rem;
  }}
  .pill-running {{ background:#eaf2f9; border:1px solid #a8c4dc; color:#1e3a5f; }}
  .pill-done    {{ background:#eef7f4; border:1px solid #a8ddc9; color:#1a5c4a; }}
  .pill-error   {{ background:#fff2f3; border:1px solid #fca5a5; color:#8b1a1a; }}
  .lbl {{
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #7b93a8;
    margin-bottom: 4px;
  }}
  .divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, #b8cddf, transparent);
    margin: 1.4rem 0;
  }}
  div[data-testid="stButton"] > button {{
    height: 26px;
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: .05em;
    text-transform: uppercase;
    border-radius: 6px;
    background: transparent;
    border: 1px solid #a8c4dc;
    color: #3a6a8f;
  }}
  div[data-testid="stButton"] > button:hover {{
    background: #eaf2f9;
    border-color: #3a6a8f;
    color: #1e3a5f;
  }}
  div[data-testid="stDownloadButton"] > button {{
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    letter-spacing: .04em;
    padding: 0.55rem 1rem;
    border-radius: 10px;
    background: #1a2838;
    border: none;
    color: #fff;
    text-transform: uppercase;
  }}
  div[data-testid="stDownloadButton"] > button:hover {{ background: #2a4560; }}
</style>
""",
    unsafe_allow_html=True,
)


# ---------- session state ----------
if "img_ab" not in st.session_state:
    st.session_state.img_ab = None
if "img_ba" not in st.session_state:
    st.session_state.img_ba = None
if "up_key_ab" not in st.session_state:
    st.session_state.up_key_ab = 0
if "up_key_ba" not in st.session_state:
    st.session_state.up_key_ba = 0


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0


def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().squeeze(0).clamp(-1.0, 1.0)
    arr = ((tensor + 1.0) * 127.5).permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@torch.inference_mode()
def run_generator(generator: torch.nn.Module, img: Image.Image, device: torch.device):
    x = preprocess_pil(img).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    y = generator(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    out = postprocess_tensor(y)
    return out, (time.perf_counter() - t0) * 1000.0


# ---------- cached functions ----------
@st.cache_resource(show_spinner=False)
def load_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CycleGAN()

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    model = model.to(device).eval()
    return model.G_ab.eval(), model.G_ba.eval(), device


@st.cache_data(show_spinner=False)
def list_examples(folder, pattern):
    return sorted(
        str(f)
        for f in Path(folder).glob(pattern)
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )


@st.cache_data(show_spinner=False)
def read_file(path):
    return Path(path).read_bytes()


@st.cache_data(show_spinner=False, max_entries=200)
def run_inference(img_bytes, direction, ckpt_path):
    gen_ab, gen_ba, device = load_model(ckpt_path)
    generator = gen_ab if direction == "ab" else gen_ba
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    out, ms = run_generator(generator, img, device)
    return pil_to_bytes(out), ms


# ---------- page header ----------
st.markdown("# Photo&thinsp;↔&thinsp;Ukiyoe", unsafe_allow_html=True)
st.caption("CycleGAN unpaired image translation · upload a ukiyoe artwork or a photo")

# ---------- checkpoint selector ----------
checkpoints = list_checkpoints()
model_ok = False
CKPT = None
if not checkpoints:
    st.error(f"No checkpoints found in {CHECKPOINTS_DIR}")
else:
    labels = [ckpt_label(c) for c in checkpoints]
    default_idx = next(
        (i for i, c in enumerate(checkpoints) if c["idf"] == 5 and c["cyc"] == 10),
        0,
    )
    idx = st.selectbox(
        "Checkpoint",
        options=list(range(len(checkpoints))),
        format_func=lambda i: labels[i],
        index=default_idx,
    )
    CKPT = checkpoints[idx]["path"]
    try:
        load_model(CKPT)
        model_ok = True
    except Exception as e:
        st.error(f"Could not load model: {e}")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


# ---------- show result ----------
def show_result(img_bytes, direction, src_lbl, tgt_lbl):
    status = st.empty()
    status.markdown('<div class="pill pill-running">🎨&ensp;Generating…</div>', unsafe_allow_html=True)
    try:
        out_bytes, ms = run_inference(img_bytes, direction, CKPT)
    except Exception as e:
        status.markdown(f'<div class="pill pill-error">✗&ensp;Error: {e}</div>', unsafe_allow_html=True)
        return

    status.markdown(f'<div class="pill pill-done">✦&ensp;Done · {ms:.0f} ms</div>', unsafe_allow_html=True)

    col1, gap, col2 = st.columns([10, 1, 10])
    with col1:
        st.markdown(f'<p class="lbl">{src_lbl}</p>', unsafe_allow_html=True)
        st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
    with gap:
        st.markdown(
            "<div style='display:flex;height:100%;align-items:center;"
            "justify-content:center;font-size:1.3rem;color:#c8b070'>→</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f'<p class="lbl">{tgt_lbl}</p>', unsafe_allow_html=True)
        st.image(Image.open(io.BytesIO(out_bytes)), use_container_width=True)

    st.download_button(
        "Download result",
        out_bytes,
        file_name=f"cyclegan_{direction}.png",
        mime="image/png",
        use_container_width=True,
    )


# ---------- example grid ----------
def show_examples(direction, pattern, state_key):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("**Examples** — click to use")

    paths = list_examples(str(EXAMPLES_DIR), pattern)
    n_cols = 6
    for row in [paths[i : i + n_cols] for i in range(0, len(paths), n_cols)]:
        cols = st.columns(n_cols)
        for col, path in zip(cols, row):
            img_bytes = read_file(path)
            with col:
                st.image(Image.open(io.BytesIO(img_bytes)), use_container_width=True)
                if st.button("Use", key=f"ex_{direction}_{path}", use_container_width=True):
                    st.session_state[state_key] = img_bytes
                    st.session_state[f"up_key_{direction}"] += 1
                    st.rerun()


# ---------- tab content ----------
def show_tab(direction, upload_label, src_lbl, tgt_lbl, pattern, state_key):
    up_key = st.session_state[f"up_key_{direction}"]
    uploaded = st.file_uploader(
        upload_label,
        type=["jpg", "jpeg", "png", "webp"],
        key=f"up_{direction}_{up_key}",
        label_visibility="collapsed",
    )
    if uploaded:
        st.session_state[state_key] = uploaded.read()

    active = st.session_state[state_key]

    if active and model_ok:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        show_result(active, direction, src_lbl, tgt_lbl)
    elif not active:
        st.markdown(
            "<div style='text-align:center;color:#c8b870;padding:1.2rem 0;"
            "font-size:0.84rem'>Upload an image or pick an example below</div>",
            unsafe_allow_html=True,
        )

    show_examples(direction, pattern, state_key)


# ---------- tabs ----------
tab_ba, tab_ab = st.tabs(["📷  Photo  ->  Ukiyoe", "🖼  Ukiyoe  ->  Photo"])

with tab_ba:
    show_tab(
        "ba",
        upload_label="Upload a photo",
        src_lbl="Photograph",
        tgt_lbl="Ukiyoe artwork",
        pattern="photo_*.jpg",
        state_key="img_ba",
    )

with tab_ab:
    show_tab(
        "ab",
        upload_label="Upload a ukiyoe artwork",
        src_lbl="Ukiyoe artwork",
        tgt_lbl="Photograph",
        pattern="ukiyoe_*.jpg",
        state_key="img_ab",
    )
