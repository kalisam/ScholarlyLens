"""
ScholarLens: Advanced Research Paper Analysis
Main application entry point
"""
# Initialize streamlit compatibility patches first
from utils.streamlit_fix import apply_fixes
apply_fixes()

import streamlit as st
from utils.environment import load_environment
from core.session import init_session_state
from ui.main_page import render_main_layout

def main():
    """Main application entry point"""
    # Set page title and configuration
    st.set_page_config(
        page_title="ScholarLens: Advanced Research Paper Analysis",
        page_icon="📑",
        layout="wide"
    )
    
    # Load environment variables
    load_environment()
    
    # Initialize session state
    init_session_state()
    
    # Render main application layout
    render_main_layout()

if __name__ == "__main__":
    main()