"""
pdf_extractor.py
----------------
Handles PDF upload and text extraction using pdfplumber + pypdf fallback.
"""

import io
import pdfplumber
from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract all text from an uploaded Streamlit PDF file object.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Extracted text as a single string, or empty string on failure.
    """
    file_bytes = uploaded_file.read()

    # Try pdfplumber first (better layout handling)
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            if pages_text:
                return "\n\n".join(pages_text)
    except Exception:
        pass

    # Fallback to pypdf
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)
    except Exception as e:
        raise ValueError(f"Could not extract text from PDF: {e}")


def get_pdf_metadata(uploaded_file) -> dict:
    """
    Extract basic metadata from a PDF file.

    Returns:
        dict with keys: num_pages, title, author
    """
    file_bytes = uploaded_file.read()
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        meta = reader.metadata or {}
        return {
            "num_pages": len(reader.pages),
            "title": meta.get("/Title", "Unknown"),
            "author": meta.get("/Author", "Unknown"),
        }
    except Exception:
        return {"num_pages": 0, "title": "Unknown", "author": "Unknown"}
