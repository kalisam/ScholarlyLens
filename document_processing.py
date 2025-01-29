import fitz  # PyMuPDF for extracting text from PDFs
import re
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file while preserving sections.
    """
    doc = fitz.open(pdf_path)
    extracted_text = []
    
    for page in doc:
        text = page.get_text("text")
        extracted_text.append(text)

    return "\n".join(extracted_text)

def clean_text(text):
    """
    Cleans extracted text by removing unnecessary spaces and special characters.
    """
    text = re.sub(r'\n+', '\n', text)  # Remove excessive newlines
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = text.strip()
    return text

def chunk_text(text, embedding_model):
    """
    Splits text into semantically meaningful chunks.
    """
    text_splitter = SemanticChunker(embedding_model)
    return text_splitter.split_text(text)

def process_pdf(pdf_path):
    """
    Full pipeline: Extracts, cleans, and chunks PDF text.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    clean_text_data = clean_text(raw_text)

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings()
    
    chunks = chunk_text(clean_text_data, embedding_model)
    
    return chunks
