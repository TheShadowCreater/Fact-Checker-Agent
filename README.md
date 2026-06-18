# 🔍 AI Fact Checker

An intelligent fact-checking web application built with **Python + Streamlit**, powered by **Groq (LLaMA 3)** for LLM inference and **Tavily** for live web search. Upload any PDF document, and the app automatically extracts verifiable factual claims, verifies them against real-time web results, and classifies each as **Verified**, **Inaccurate**, or **False** — complete with confidence scores, reasoning, and source links.

---

## ✨ Features

- 📄 **PDF Upload** — drag-and-drop support for any text-based PDF
- 🧠 **AI Claim Extraction** — LLaMA 3 (via Groq) identifies statistics, dates, financial figures, and technical statements
- 🌐 **Live Web Verification** — each claim is verified against real-time Tavily search results
- 🏷️ **Verdict Classification** — Verified / Inaccurate / False with confidence scores (0–100%)
- 📊 **Executive Dashboard** — summary metrics and interactive Plotly charts
- 🔗 **Source Links** — evidence URLs for every verified or disputed claim
- ⬇️ **CSV Export** — full downloadable report
- 🔄 **Progress Indicators** — real-time status for each processing step
- 🎛️ **Filter & Search** — filter results by verdict or search claim text
- 🔒 **No API key UI** — credentials are managed via environment variables only

---

## 🗂️ Project Structure

```
fact-checker/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore
├── README.md
└── src/
    ├── __init__.py
    ├── pdf_extractor.py    # PDF text extraction (pdfplumber + pypdf)
    ├── claim_extractor.py  # Factual claim extraction via Groq (LLaMA 3)
    ├── fact_verifier.py    # Live web verification via Tavily + Groq
    ├── report_generator.py # CSV report + summary stats builder
    └── charts.py           # Plotly chart builders
```

---

## ⚙️ Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Your Groq API key | ✅ Yes |
| `TAVILY_API_KEY` | Your Tavily API key | ✅ Yes |

- Get your Groq key at [console.groq.com](https://console.groq.com)
- Get your Tavily key at [app.tavily.com](https://app.tavily.com)

> **Security note:** API keys are loaded from environment variables at startup. They are never displayed in the UI, never passed to the browser, and never asked of the user.

---

## 🚀 Local Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fact-checker.git
cd fact-checker
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
GROQ_API_KEY=gsk_your_groq_key_here
TAVILY_API_KEY=tvly-your_tavily_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Deployment on Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/fact-checker.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repository
4. Set **Main file path** to `app.py`
5. Click **Advanced settings → Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_groq_key_here"
TAVILY_API_KEY = "tvly-your_tavily_key_here"
```

6. Click **Deploy**

---

## 📋 How It Works

1. **Upload PDF** — Text is extracted using `pdfplumber` (with `pypdf` as fallback).
2. **Claim Extraction** — Groq (LLaMA 3.3 70B) reads the document and identifies claims containing specific numbers, dates, percentages, financial data, or technical assertions.
3. **Live Verification** — Each claim is sent to Tavily for a real-time web search. The top results are then analysed by Groq, which compares the evidence against the claim.
4. **Classification** — Each claim is labelled:
   - ✅ **Verified** — supported by credible sources
   - ⚠️ **Inaccurate** — contains errors or outdated information
   - ❌ **False** — contradicted by credible sources
5. **Report** — Results are displayed with charts and can be downloaded as CSV.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | LLaMA 3.3 70B via Groq |
| Web Search | Tavily Search API |
| PDF Parsing | pdfplumber, pypdf |
| Charts | Plotly |
| Data | pandas |
| Config | python-dotenv |

---

## 📝 Notes

- The app works best with **text-based PDFs** (reports, articles, whitepapers). Scanned/image PDFs may not extract correctly.
- Processing time depends on the number of claims (~3–10 seconds per claim for web verification).
- Document text is truncated to 12,000 characters to stay within token limits.
- The maximum number of claims is adjustable in the sidebar (5–30).
- Groq's free tier is generous; Tavily offers 1,000 free searches/month.
