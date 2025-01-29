import streamlit as st
import tempfile
from document_processing import process_pdf
from langchain_community.vectorstores import FAISS

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Upload PDF
upload_pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if upload_pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(upload_pdf_file.getvalue())
        temp_pdf_path = temp_file.name

    try:
        # Process PDF
        chunks = process_pdf(temp_pdf_path)
        
        # Create vector store
        st.session_state.vector_store = FAISS.from_texts(chunks)
        st.success("PDF processed successfully!")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

st.info("Upload a PDF to start the chatbot.")
