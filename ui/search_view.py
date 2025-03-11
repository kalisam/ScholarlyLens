"""
Academic paper search view for ScholarLens
"""
import streamlit as st
from core.search_service import search_papers, display_paper_results

def render_search_view():
    """Render the academic search interface"""
    # Search for papers
    st.header("Search for Academic Papers")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter search terms:")
    with col2:
        source = st.selectbox("Source:", ["arXiv", "Semantic Scholar"])
    
    if st.button("Search") and search_query:
        st.session_state.search_results = search_papers(source, search_query)
        display_paper_results(st.session_state.search_results, source)