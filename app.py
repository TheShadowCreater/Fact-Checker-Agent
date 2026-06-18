"""
app.py
------
Main Streamlit application for the AI Fact-Checker.
Uses Groq (LLaMA 3) for LLM inference and Tavily for live web search.
API keys are loaded from environment variables — no user input required.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

# ── load env vars before anything else ───────────────────────────────────────
load_dotenv()

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pdf_extractor import extract_text_from_pdf, get_pdf_metadata
from claim_extractor import extract_claims
from fact_verifier import verify_claim
from report_generator import build_dataframe, to_csv_bytes, compute_summary
from charts import pie_chart, confidence_bar_chart, category_breakdown_chart

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fact Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #0f172a;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] { background: #1e293b; }
    [data-testid="stMetric"] {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #334155;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.82rem; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 2rem; }
    .badge-verified   { background:#14532d; color:#4ade80; border:1px solid #16a34a;
                        padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-inaccurate { background:#451a03; color:#fb923c; border:1px solid #c2410c;
                        padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .badge-false      { background:#450a0a; color:#f87171; border:1px solid #dc2626;
                        padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
    .claim-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .claim-text { font-size: 1rem; font-weight: 500; color: #f1f5f9; margin-bottom: 8px; }
    .claim-meta { font-size: 0.82rem; color: #94a3b8; }
    .reasoning  { font-size: 0.88rem; color: #cbd5e1; margin-top: 8px; line-height: 1.6; }
    .source-link{ font-size: 0.82rem; color: #60a5fa; text-decoration: none; }
    .stProgress > div > div > div { background: #6366f1; border-radius: 4px; }
    .stButton > button {
        background: linear-gradient(135deg,#6366f1,#8b5cf6);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: 600; font-size: 0.95rem;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: .85; }
    [data-testid="stFileUploader"] {
        background: #1e293b; border: 2px dashed #334155;
        border-radius: 12px; padding: 12px;
    }
    hr { border-color: #334155; }
    .stTabs [data-baseweb="tab-list"] { background: #1e293b; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 8px; }
    .stTabs [aria-selected="true"] { background: #6366f1 !important; color: #fff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── runtime key validation (server-side only) ─────────────────────────────────

def _check_keys() -> list[str]:
    """Return a list of missing required environment variable names."""
    missing = []
    if not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.environ.get("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    return missing


# ── helpers ───────────────────────────────────────────────────────────────────

def verdict_badge(verdict: str) -> str:
    cls = {
        "Verified": "badge-verified",
        "Inaccurate": "badge-inaccurate",
        "False": "badge-false",
    }.get(verdict, "badge-inaccurate")
    icon = {"Verified": "✅", "Inaccurate": "⚠️", "False": "❌"}.get(verdict, "❓")
    return f'<span class="{cls}">{icon} {verdict}</span>'


def confidence_color(score: float) -> str:
    if score >= 0.75:
        return "#22c55e"
    if score >= 0.5:
        return "#f59e0b"
    return "#ef4444"


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 AI Fact Checker")
    st.markdown("*Verify document claims with live web search*")
    st.divider()

    st.markdown("#### ⚙️ Settings")
    max_claims = st.slider("Max claims to extract", 5, 30, 15, 1)
    show_snippets = st.checkbox("Show source snippets", value=True)

    st.divider()
    st.markdown(
        """
        **How it works:**
        1. Upload a PDF document
        2. LLaMA 3 extracts factual claims
        3. Each claim is verified via live web search
        4. Results shown with verdict + confidence
        5. Download full CSV report
        """,
    )
    st.divider()
    st.markdown(
        "<div style='color:#475569; font-size:0.78rem;'>Powered by Groq · Tavily · Streamlit</div>",
        unsafe_allow_html=True,
    )

# ── main header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="text-align:center; padding: 32px 0 16px;">
        <h1 style="font-size:2.6rem; font-weight:700; color:#f1f5f9; margin:0;">
            🔍 AI Fact Checker
        </h1>
        <p style="color:#94a3b8; font-size:1.05rem; margin-top:8px;">
            Upload a PDF · Extract claims · Verify with live web search
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── key check (server-side, no keys shown in UI) ──────────────────────────────

missing_keys = _check_keys()
if missing_keys:
    st.error(
        f"⚠️ Server configuration error: the following environment variables are not set: "
        f"`{'`, `'.join(missing_keys)}`. "
        f"Please add them to your `.env` file and restart the app."
    )
    st.stop()

# ── file upload ───────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Drop your PDF here",
    type=["pdf"],
    help="Upload any text-based PDF — reports, articles, research papers, whitepapers.",
)

if uploaded_file is None:
    st.info("👆 Upload a PDF to get started.")
    st.stop()

# ── run analysis ──────────────────────────────────────────────────────────────

run_btn = st.button("🚀 Analyse Document", use_container_width=True)

if "results" not in st.session_state:
    st.session_state.results = []
    st.session_state.summary = {}
    st.session_state.df = None
    st.session_state.pdf_meta = {}

if run_btn:
    st.session_state.results = []
    st.session_state.summary = {}
    st.session_state.df = None

    # Step 1 — extract text
    with st.status("📄 Extracting text from PDF…", expanded=True) as status:
        try:
            uploaded_file.seek(0)
            text = extract_text_from_pdf(uploaded_file)
            uploaded_file.seek(0)
            meta = get_pdf_metadata(uploaded_file)
            st.session_state.pdf_meta = meta
            st.write(f"✅ Extracted text from **{meta['num_pages']} pages**")
        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
            st.stop()

        if not text.strip():
            st.error("No text could be extracted. The PDF may be scanned/image-based.")
            st.stop()

        # Step 2 — extract claims
        st.write("🧠 Identifying factual claims…")
        try:
            claims = extract_claims(text, max_claims=max_claims)
            st.write(f"✅ Found **{len(claims)} verifiable claims**")
        except Exception as e:
            st.error(f"Claim extraction failed: {e}")
            st.stop()

        if not claims:
            st.warning("No verifiable factual claims found in the document.")
            st.stop()

        # Step 3 — verify each claim
        st.write("🌐 Verifying claims with live web search…")
        progress = st.progress(0)
        results = []

        for i, claim in enumerate(claims):
            try:
                verification = verify_claim(claim["claim"])
            except Exception as e:
                verification = {
                    "verdict": "Inaccurate",
                    "confidence": 0.3,
                    "reasoning": f"Verification error: {e}",
                    "sources": [],
                }
            results.append({**claim, **verification})
            progress.progress((i + 1) / len(claims))

        summary = compute_summary(results)
        df = build_dataframe(results)

        st.session_state.results = results
        st.session_state.summary = summary
        st.session_state.df = df

        status.update(label="✅ Analysis complete!", state="complete", expanded=False)

# ── display results ───────────────────────────────────────────────────────────

if not st.session_state.results:
    st.stop()

results = st.session_state.results
summary = st.session_state.summary
df = st.session_state.df
meta = st.session_state.pdf_meta

st.divider()

# Executive summary metrics
st.markdown("### 📊 Executive Summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Claims", summary["total"])
c2.metric("✅ Verified", summary["verified"])
c3.metric("⚠️ Inaccurate", summary["inaccurate"])
c4.metric("❌ False", summary["false"])
c5.metric("Avg Confidence", f"{summary['avg_confidence']:.0%}")

st.divider()

# Charts
tab1, tab2, tab3 = st.tabs(["📈 Verdict Distribution", "📊 Confidence Scores", "🏷️ By Category"])

with tab1:
    st.plotly_chart(pie_chart(summary), use_container_width=True)

with tab2:
    if df is not None and not df.empty:
        st.plotly_chart(confidence_bar_chart(df), use_container_width=True)

with tab3:
    if df is not None and not df.empty:
        st.plotly_chart(category_breakdown_chart(df), use_container_width=True)

st.divider()

# Download report
if df is not None:
    csv_bytes = to_csv_bytes(df)
    st.download_button(
        label="⬇️ Download CSV Report",
        data=csv_bytes,
        file_name="fact_check_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# Claim-by-claim results
st.markdown("### 🔎 Claim-by-Claim Results")

col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    verdict_filter = st.multiselect(
        "Filter by verdict",
        ["Verified", "Inaccurate", "False"],
        default=["Verified", "Inaccurate", "False"],
    )
with col_f2:
    search_term = st.text_input("Search claims", placeholder="Type to filter…")

filtered = [
    r for r in results
    if r["verdict"] in verdict_filter
    and (not search_term or search_term.lower() in r["claim"].lower())
]

st.caption(f"Showing {len(filtered)} of {len(results)} claims")

for r in filtered:
    conf = r["confidence"]
    conf_pct = f"{conf:.0%}"
    conf_col = confidence_color(conf)

    sources_html = ""
    if r.get("sources"):
        links = " &nbsp;·&nbsp; ".join(
            f'<a class="source-link" href="{s}" target="_blank">🔗 Source {i+1}</a>'
            for i, s in enumerate(r["sources"][:3])
        )
        sources_html = f"<div style='margin-top:8px'>{links}</div>"

    snippet_html = ""
    if show_snippets and r.get("source_snippet"):
        snippet_html = f"""
        <div style="margin-top:8px; padding:8px 12px; background:#0f172a;
                    border-left:3px solid #334155; border-radius:4px;
                    font-size:0.82rem; color:#94a3b8; font-style:italic;">
            "{r['source_snippet'][:200]}{'…' if len(r['source_snippet']) > 200 else ''}"
        </div>
        """

    st.markdown(
        f"""
        <div class="claim-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                <div class="claim-text">#{r['id']} · {r['claim']}</div>
                <div style="display:flex; gap:10px; align-items:center; flex-shrink:0;">
                    {verdict_badge(r['verdict'])}
                    <span style="color:{conf_col}; font-weight:600; font-size:0.88rem;">{conf_pct}</span>
                </div>
            </div>
            <div class="claim-meta">Category: <b>{r['category']}</b></div>
            {snippet_html}
            <div class="reasoning">💬 {r['reasoning']}</div>
            {sources_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="text-align:center; color:#475569; font-size:0.8rem; padding:32px 0 16px;">
        Powered by Groq · Tavily · Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
