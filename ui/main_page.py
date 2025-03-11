"""
Main layout for ScholarLens UI
"""
import streamlit as st
from streamlit_option_menu import option_menu
from ui.upload_view import render_upload_view
from ui.search_view import render_search_view
from ui.batch_view import render_batch_view
from ui.settings_view import render_settings_view
from core.pdf_processor import process_uploaded_pdf

def render_main_layout():
    """
    Render the main application layout with navigation and views.
    """
    st.title("ScholarLens: Advanced Research Paper Analysis")
    
    # Create sidebar navigation
    with st.sidebar:
        selected_tab = option_menu(
            "Navigation",
            ["Upload PDF", "Search Papers", "Batch Process", "Settings"],
            icons=["file-earmark-pdf", "search", "list-task", "gear"],
            default_index=0
        )
        
        # Store selected tab in session state
        st.session_state.current_tab = selected_tab
        
        # Add info about the app
        st.divider()
        st.info(
            "ScholarLens helps researchers analyze academic papers, "
            "identify research gaps, and explore citation networks."
        )
    
    # Process selected paper if available
    if st.session_state.selected_paper:
        with st.spinner(f"Processing paper: {st.session_state.selected_paper['title']}"):
            process_uploaded_pdf(st.session_state.selected_paper['content'])
            # Reset selected paper to avoid reprocessing
            st.session_state.selected_paper = None
    
    # Render the selected view
    if st.session_state.current_tab == "Upload PDF":
        render_upload_view()
    elif st.session_state.current_tab == "Search Papers":
        render_search_view()
    elif st.session_state.current_tab == "Batch Process":
        render_batch_view()
    elif st.session_state.current_tab == "Settings":
        render_settings_view()