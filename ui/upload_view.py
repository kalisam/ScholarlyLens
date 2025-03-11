"""
PDF upload view for ScholarLens
"""
import streamlit as st
from core.pdf_processor import process_uploaded_pdf

def render_upload_view():
    """Render the PDF upload interface"""
    # Upload PDF
    upload_pdf_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

    if upload_pdf_file:
        process_uploaded_pdf(upload_pdf_file)