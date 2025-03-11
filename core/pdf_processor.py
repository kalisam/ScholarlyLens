"""
PDF processing functionality for ScholarLens
"""
import streamlit as st
from typing import Dict, List, Any, Optional, Union, BinaryIO
from ui.components.analysis_tabs import display_analysis_tabs

def process_uploaded_pdf(pdf_content: Union[bytes, BinaryIO]) -> bool:
    """
    Process an uploaded or downloaded PDF file using improved pipeline.
    
    Args:
        pdf_content: PDF file content as bytes or file-like object
        
    Returns:
        Boolean indicating success or failure
    """
    try:
        # Convert to bytes if a file-like object was passed
        if hasattr(pdf_content, 'read'):
            pdf_content = pdf_content.read()
        
        # Show processing status
        status_placeholder = st.empty()
        status_placeholder.info("Processing PDF - Step 1/5: Extracting text and media...")
        
        # Process the PDF with improved extraction
        if st.session_state.config.extract_images or st.session_state.config.extract_tables:
            # Use PyMuPDF for enhanced extraction
            docs, figures, tables = st.session_state.processor.load_pdf_with_pymupdf(pdf_content)
            st.session_state.figures = figures
            st.session_state.tables = tables
        else:
            # Use standard extraction
            docs = st.session_state.processor.load_pdf(pdf_content)
            st.session_state.figures = []
            st.session_state.tables = []
        
        status_placeholder.info("Processing PDF - Step 2/5: Creating semantic chunks...")
        chunks = st.session_state.processor.create_chunks(docs, st.session_state.rag_pipeline.embeddings)
        
        status_placeholder.info("Processing PDF - Step 3/5: Building vector index...")
        # Create vector store
        st.session_state.rag_pipeline.create_vector_store(chunks)
        
        status_placeholder.info("Processing PDF - Step 4/5: Analyzing document structure...")
        # Extract sections and full text - use improved version if available
        if hasattr(st.session_state.processor, 'extract_sections_improved'):
            sections = st.session_state.processor.extract_sections_improved(docs)
        else:
            sections = st.session_state.processor.extract_sections(docs)
        
        full_text = " ".join([doc.page_content for doc in docs])
        
        status_placeholder.info("Processing PDF - Step 5/5: Identifying references...")
        # Extract top references
        top_references = st.session_state.reference_analyzer.get_most_cited_references(full_text)
        st.session_state.top_references = top_references
        
        # Generate citation network
        if top_references:
            paper_title = sections.get('title', 'Untitled Paper')
            if not paper_title or paper_title == 'Untitled Paper':
                # Try to extract title from first page
                first_page = docs[0].page_content if docs else ""
                lines = first_page.split('\n')
                if lines and lines[0].strip():
                    paper_title = lines[0].strip()
            
            # Build citation network
            st.session_state.citation_network.build_network_from_references(
                paper_title, top_references
            )
        
        # Generate paper summary if not cached
        summary = st.session_state.rag_pipeline.generate_paper_summary(
            sections['abstract'] + sections['introduction'] + sections['conclusion']
        )
        st.session_state.summary = summary
        
        # Generate paper recommendations
        recommendations = st.session_state.rag_pipeline.generate_paper_recommendations(
            sections['abstract'] + sections['introduction'] + sections['conclusion']
        )
        st.session_state.recommendations = recommendations
        
        # Clear status message
        status_placeholder.empty()
        
        # Display analysis tabs
        display_analysis_tabs(sections, full_text)
        
        return True
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False