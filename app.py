"""app.py — redesigned Streamlit UI for Multimodal Clinical AI
Provides a clean, modern single-image / single-query interface
and improved visualization tabs (Results, Explainability, Retrieval, Report).

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

APP_DIR = Path(__file__).parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
from src.pipeline.orchestrator import MultimodalClinicalOrchestrator
from src.retrieval.faiss_retrieval import MultimodalFAISSRetriever


st.set_page_config(page_title="Clinical AI Dashboard", page_icon="🏥", layout="wide")


BASE_CSS = """
<style>
:root{
    --bg:#04060a; --card:#071429; --card2:#08182b; --muted:#9fb7d0;
    --neon1:#7c3aed; --neon2:#06b6d4; --neon3:#22c55e; --neon4:#ff6b6b;
}
html,body{background:
    radial-gradient(circle at top left, rgba(139,92,246,0.24), transparent 35%),
    radial-gradient(circle at top right, rgba(34,211,238,0.22), transparent 28%),
    linear-gradient(180deg,#06111f 0%, #0a1730 60%, #060b17 100%);}
body{font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif; color:#e8f2ff}
.block-container{padding-top:1rem; padding-bottom:2rem; max-width: 1580px;}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:28px 28px;border-radius:26px;background:linear-gradient(135deg, rgba(139,92,246,0.22), rgba(34,211,238,0.12));box-shadow:0 18px 45px rgba(2,6,23,0.55);margin-bottom:24px;border:1px solid rgba(255,255,255,0.08);backdrop-filter: blur(12px)}
.brand{display:flex;align-items:center;gap:16px}
.brand .logo{width:64px;height:64px;border-radius:18px;background:linear-gradient(135deg,var(--accent1),var(--accent2));display:flex;align-items:center;justify-content:center;font-weight:900;color:white;font-size:1.1rem;box-shadow:0 14px 35px rgba(139,92,246,0.35)}
.brand .title{font-size:1.65rem;font-weight:900;letter-spacing:-0.02em}
.header-sub{font-size:0.98rem;color:var(--muted);margin-top:4px}
.hero{background:linear-gradient(135deg, rgba(124,58,237,0.06), rgba(6,182,212,0.04));border:1px solid rgba(124,58,237,0.08);border-radius:30px;padding:26px 28px;box-shadow:0 30px 80px rgba(7,14,30,0.6);margin-bottom:22px;overflow:hidden;position:relative}
.hero:after{content:"";position:absolute;inset:auto -20px -20px auto;width:240px;height:240px;border-radius:50%;background:radial-gradient(circle, rgba(124,58,237,0.22), transparent 55%);filter:blur(6px);transform:translate(8px,4px)}
.hero-title{font-size:2.35rem;font-weight:900;letter-spacing:-0.04em;line-height:1.03;margin-bottom:8px;color:var(--neon1);text-shadow:0 6px 28px rgba(124,58,237,0.18)}
.hero-desc{font-size:1.02rem;color:#bcdff6;max-width:900px;line-height:1.6}
.hero-pills{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}
.pill{padding:0.6rem 1rem;border-radius:999px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.03);font-size:0.86rem;font-weight:800;color:#e6f6ff}
.pill.orange{background:linear-gradient(90deg, rgba(255,107,107,0.06), rgba(255,107,107,0.03));border-color:rgba(255,107,107,0.12)}
.pill.cyan{background:linear-gradient(90deg, rgba(6,182,212,0.06), rgba(6,182,212,0.03));border-color:rgba(6,182,212,0.12)}
.pill.green{background:linear-gradient(90deg, rgba(34,197,94,0.06), rgba(34,197,94,0.03));border-color:rgba(34,197,94,0.12)}
.card,.report-box,.visual-panel{background:linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));border:1px solid rgba(255,255,255,0.08);border-radius:24px;padding:24px;box-shadow:0 18px 45px rgba(2,6,23,0.35)}
.pred-card{display:flex;gap:18px;align-items:center;justify-content:space-between;min-height:140px}
.pred-main{font-size:2rem;font-weight:900;color:#f8fbff;line-height:1.05}
.pred-sub{font-size:1rem;color:var(--muted);margin-top:8px}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px;margin-top:18px}
.metric-box{background:linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));border-radius:22px;padding:24px;text-align:center;min-height:138px;border:1px solid rgba(255,255,255,0.10);position:relative;overflow:hidden}
.metric-box:before{content:"";position:absolute;inset:-2px auto auto -2px;width:90px;height:90px;border-radius:50%;background:radial-gradient(circle, rgba(139,92,246,0.30), transparent 70%)}
.metric-value{font-size:2.3rem;font-weight:900;color:#bff4ff}
.metric-label{font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em}
.section-title{font-size:0.82rem;font-weight:800;color:#d6e7ff;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:12px}
.report-box{line-height:1.8;color:#dbeeff;min-height:220px}
.visual-panel{padding:18px}
img{max-width:100%;height:auto;border-radius:18px}
.stButton>button{border-radius:16px;padding:0.95rem 1.25rem;font-weight:900;background:linear-gradient(90deg,var(--neon1),var(--neon2));color:white;border:none;box-shadow:0 18px 40px rgba(124,58,237,0.18);transform:translateZ(0);}
.stButton>button:hover{box-shadow:0 28px 70px rgba(6,182,212,0.22);transform:translateY(-2px)}
.stDownloadButton>button{border-radius:16px;padding:0.75rem 1rem;font-weight:700}
div[data-testid="stTabs"] button{font-size:1rem;padding:0.7rem 1rem}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"] > div,.stSlider [role="slider"]{border-radius:14px}
/* responsive tweaks */
@media (max-width:640px){.topbar{flex-direction:column;align-items:flex-start}.brand .title{font-size:1.3rem}.pred-card{flex-direction:column;align-items:flex-start}}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MultimodalClinicalOrchestrator(config, device=device)


# Helper visuals
def _plot_shap(tokens: List[dict]):
    labels = [t.get("token", "?") for t in tokens[:15]]
    values = [t.get("shap_score", 0.0) for t in tokens[:15]]
    colors = ["#16a34a" if v >= 0 else "#ef4444" for v in values]
    fig = go.Figure(go.Bar(x=values[::-1], y=labels[::-1], orientation="h", marker_color=colors[::-1]))
    fig.update_layout(margin=dict(l=10, r=10, t=8, b=30), height=360, paper_bgcolor="white")
    return fig


def _plot_donut(img_pct, text_pct, dominant):
    fig = go.Figure(go.Pie(values=[img_pct, text_pct], labels=["Image", "Text"], hole=0.62, marker_colors=["#22c55e", "#38bdf8"]))
    fig.update_layout(margin=dict(l=0, r=0, t=6, b=6), height=260, showlegend=False)
    return fig


def _plot_rag(retrieved: List[dict]):
    if not retrieved:
        return None
    names = [r.get("image_id", "?")[:20] for r in retrieved]
    sims = [r.get("similarity", 0.0) for r in retrieved]
    fig = go.Figure(go.Bar(x=names, y=sims, marker_color="#0ea5a4"))
    fig.update_layout(margin=dict(l=10, r=10, t=6, b=40), height=300, paper_bgcolor="white")
    return fig


def _make_overlay(img: Image.Image, cam_map: np.ndarray):
    try:
        import matplotlib

        img_arr = np.array(img.resize((224, 224))).astype(np.float32)
        cam = np.zeros((224, 224)) if cam_map is None else cam_map
        heat = (matplotlib.colormaps.get_cmap("jet")(cam)[:, :, :3] * 255).astype(np.float32)
        overlay = np.clip(0.55 * img_arr + 0.45 * heat, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    except Exception:
        return img


# Sidebar controls
with st.sidebar:
    st.markdown("## 🏥 Clinical AI")
    st.markdown("Single image + single query mode")
    uploaded = st.file_uploader("Upload medical scan", type=["png", "jpg", "jpeg"])
    clinical_text = st.text_area(
        "Clinical query / notes",
        value=(
            "45 year-old man with a past history of malignant testicular neoplasm and Hashimoto's thyroiditis, "
            "presents with worsening eye swelling, dryness, lid retraction and intermittent blurry vision."
        ),
        height=145,
        placeholder="Enter one clinical question or note...",
        )
    faiss_k = st.slider("Retrieved cases (K)", 1, 10, 5)
    run = st.button("Run Analysis", width="stretch")


st.header("Clinical AI Dashboard")
st.write("Streamlined visual interface — Predictions, Explainability, Retrieval, and Report.")

st.markdown(
        """
<div class="topbar">
    <div class="brand">
        <div class="logo">AI</div>
        <div>
            <div class="title">Clinical AI Studio</div>
            <div class="header-sub">A bold oversized interface for multimodal analysis, XAI, retrieval, and clinical reporting</div>
        </div>
    </div>
    <div style="text-align:right;min-width:180px;">
        <div style="font-size:0.9rem;color:#dbeeff;margin-bottom:6px;">Mode: <strong style="color:#bff4ff">Visual Studio</strong></div>
        <div style="font-size:0.78rem;color:#9eb0c7">Large cards · large charts · large panels</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
)

st.markdown(
        """
<div class="hero">
    <div class="hero-title">A bold clinical AI stage for presentations</div>
    <div class="hero-desc">This version is designed to feel different from a standard dashboard: oversized panels, dramatic gradients, stronger spacing, and a presentation-ready layout for live class demos.</div>
    <div class="hero-pills">
        <div class="pill cyan">Scan intelligence</div>
        <div class="pill green">Explainability engine</div>
        <div class="pill orange">Retrieval gallery</div>
        <div class="pill">Clinical report export</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
)


if run:
    if uploaded is None:
        st.error("Please upload one medical scan before running.")
        st.stop()
    if not clinical_text or not clinical_text.strip():
        st.error("Please provide a clinical query or notes.")
        st.stop()

    # persist uploaded file to disk for orchestrator
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        img = Image.open(tmp_path).convert("RGB")
    except Exception:
        st.error("Uploaded file could not be read as an image.")
        st.stop()

    with st.spinner("Loading pipeline..."):
        orchestrator = load_pipeline()

    progress = st.progress(0)
    try:
        progress.progress(10)
        results = orchestrator.process_case(tmp_path, clinical_text)
        progress.progress(80)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        raise
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        progress.progress(100)

    # Extract results safely
    pred_label = results.get("pred_label", "Unknown")
    confidence = float(results.get("confidence", 0.0))
    cam_map = results.get("gradcam_map", None)
    shap_tokens = results.get("top_shap_tokens", [])
    dominant = results.get("dominant_modality", "image")
    img_pct = float(results.get("image_contribution_pct", 50.0))
    text_pct = float(results.get("text_contribution_pct", 50.0))
    retrieved = results.get("retrieved_cases", [])
    final_report = results.get("final_report", "")

    # Top row: prediction card + quick metrics
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='header'><div class='app-title'>Prediction: <span style='color:#0f172a'>{pred_label}</span></div></div>")
        st.markdown(f"<div class='small muted'>Confidence: <strong>{confidence:.1%}</strong></div>")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.plotly_chart(_plot_donut(img_pct, text_pct, dominant), width="stretch", key="prediction_donut_top")
    with c3:
        st.metric(label="Retrieved (K)", value=len(retrieved), delta=f"Top {faiss_k}")

    tabs = st.tabs(["Scan & Grad-CAM", "Explainability", "Retrieval (RAG)", "Clinical Report"])

    with tabs[0]:
        lcol, rcol = st.columns([1, 1], gap="large")
        # interactive image + overlay using Plotly for zoom/pan and adjustable opacity
        cam_opacity = st.slider("Grad-CAM opacity", 0.0, 1.0, 0.55)
        img_arr = np.array(img.convert("RGB"))
        # normalize cam_map to 0..1 if present
        cam_norm = None
        if cam_map is not None:
            try:
                cm = np.array(cam_map, dtype=np.float32)
                cm -= cm.min()
                if cm.max() > 0:
                    cm = cm / (cm.max())
                cam_norm = cm
            except Exception:
                cam_norm = None

        # Plotly figure with image and optional heatmap overlay
        fig = go.Figure()
        fig.add_trace(go.Image(z=img_arr))
        if cam_norm is not None:
            fig.add_trace(
                go.Heatmap(z=cam_norm, colorscale="Jet", opacity=cam_opacity, showscale=False, zmin=0, zmax=1)
            )
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=520)
        with lcol:
            st.subheader("Original Scan (interactive)")
            st.plotly_chart(fig, width="stretch", key="scan_interactive_plot")
        with rcol:
            st.subheader("Grad-CAM Overlay Preview")
            # Provide overlay image fallback
            overlay = _make_overlay(img, cam_map)
            st.image(overlay, width="stretch")

    with tabs[1]:
        st.subheader("SHAP (Text) Importance")
        if shap_tokens:
            try:
                # tokens might be list of dicts with 'token' and 'shap_score'
                df_shap = pd.DataFrame([{"token": t.get("token", "?"), "score": t.get("shap_score", 0.0)} for t in shap_tokens])
                c1s, c2s = st.columns([2, 1])
                with c1s:
                    st.plotly_chart(_plot_shap(shap_tokens), width="stretch", key="shap_importance_plot")
                    st.markdown("**Token details**")
                    st.dataframe(df_shap, width="stretch")
                with c2s:
                    sel = st.selectbox("Select token", options=list(df_shap["token"][:30]))
                    sel_row = df_shap[df_shap["token"] == sel].head(1)
                    if not sel_row.empty:
                        st.write(f"Token: **{sel}**")
                        st.write(f"SHAP score: {float(sel_row['score'].iloc[0]):.4f}")
            except Exception:
                st.write("Failed to render SHAP chart or table.")
        else:
            st.info("SHAP values are not available for this case.")

        st.subheader("Modality Contribution")
        st.plotly_chart(_plot_donut(img_pct, text_pct, dominant), width="stretch", key="modality_donut_plot")

    with tabs[2]:
        st.subheader("Retrieved Similar Cases")
        if retrieved:
            df_rows = []
            for r in retrieved:
                df_rows.append({
                    "Case": r.get("image_id", "?"),
                    "Similarity": f"{r.get('similarity', 0):.2%}",
                    "Label": r.get("label_name", "?")
                })
            st.table(pd.DataFrame(df_rows))
            fig = _plot_rag(retrieved)
            if fig:
                st.plotly_chart(fig, width="stretch", key="rag_similarity_plot")
        else:
            st.info("No retrieved cases (FAISS index missing or empty).")

    with tabs[3]:
        st.subheader("Agent-Synthesized Report")
        if final_report:
            st.markdown(f"<div class='card'>{final_report}</div>", unsafe_allow_html=True)
            # download options
            st.markdown("---")
            txt = final_report if isinstance(final_report, str) else str(final_report)
            st.download_button("Download report (TXT)", txt, file_name="clinical_report.txt", mime="text/plain")
            st.download_button("Download report (MD)", txt, file_name="clinical_report.md", mime="text/markdown")
            try:
                # if reportlab is available, generate simple PDF
                from io import BytesIO
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas

                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                text_obj = c.beginText(40, 730)
                for line in txt.splitlines():
                    text_obj.textLine(line)
                c.drawText(text_obj)
                c.showPage()
                c.save()
                buffer.seek(0)
                st.download_button("Download report (PDF)", buffer, file_name="clinical_report.pdf", mime="application/pdf")
            except Exception:
                st.info("PDF export not available (install reportlab to enable).")
        else:
            st.info("No clinical report produced for this run.")

    st.caption("Redesigned UI — let me know any color/layout preferences or extra charts to add.")

