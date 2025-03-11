"""
Batch processing service for ScholarLens
"""
import streamlit as st
import os
import base64

def process_batch_files(uploaded_files):
    """
    Process multiple PDF files with batch processing.
    
    Args:
        uploaded_files: List of uploaded file objects
    """
    # Start batch processor if not already running
    if not st.session_state.batch_manager.batch_processor or \
        not st.session_state.batch_manager.batch_processor.running:
        st.session_state.batch_manager.start_batch_processor(
            num_workers=st.session_state.config.batch_workers
        )
    
    # Process each file
    result_ids = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Adding file {i+1}/{len(uploaded_files)}: {file.name}")
        progress_bar.progress((i + 0.5) / len(uploaded_files))
        
        # Get file content
        pdf_content = file.getvalue()
        
        # Add to batch processor
        result_id = st.session_state.batch_manager.batch_processor.add_paper(
            pdf_content, file.name
        )
        result_ids.append(result_id)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text(f"Added {len(uploaded_files)} files to processing queue")
    progress_bar.empty()
    
    st.success(f"Started processing {len(uploaded_files)} files in the background")
    
    return result_ids

def get_batch_status():
    """
    Get the current status of batch processing.
    
    Returns:
        Dictionary with batch status information
    """
    return st.session_state.batch_manager.get_batch_status()

def export_batch_results(format="json"):
    """
    Export batch results to a file.
    
    Args:
        format: Export format ("json", "csv", "zip")
        
    Returns:
        Path to the exported file, or None if failed
    """
    try:
        return st.session_state.batch_manager.save_results(format=format)
    except Exception as e:
        st.error(f"Error exporting results: {str(e)}")
        return None

def clear_batch_results():
    """Clear all batch processing results."""
    if st.session_state.batch_manager.batch_processor:
        st.session_state.batch_manager.batch_processor.clear_results()
        return True
    return False