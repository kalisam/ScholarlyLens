"""
Session state management for ScholarLens
"""
import streamlit as st
import os
from config import ModelConfig
from document_processing import DocumentProcessor
from academic_apis import AcademicAPIs
from reference_analyzer import ReferenceAnalyzer
from improvements.citation_network import CitationNetwork
from improvements.batch_processing import BatchAnalysisManager
from improvements.multi_model import ModelFactory

# Break circular import by importing RAGPipeline conditionally
def get_rag_pipeline():
    from rag_functions import RAGPipeline
    return RAGPipeline

def init_session_state():
    """Initialize session state variables."""
    # Initialize config
    if 'config' not in st.session_state:
        st.session_state.config = ModelConfig()
        # Check for API keys in environment variables
        if os.environ.get("OPENAI_API_KEY"):
            st.session_state.config.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if os.environ.get("ANTHROPIC_API_KEY"):
            st.session_state.config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if os.environ.get("GOOGLE_API_KEY"):
            st.session_state.config.google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor(st.session_state.config)
    
    # Initialize RAG pipeline if API key is available for selected provider
    if 'rag_pipeline' not in st.session_state and has_valid_api_key(st.session_state.config):
        try:
            RAGPipeline = get_rag_pipeline()
            st.session_state.rag_pipeline = RAGPipeline(st.session_state.config)
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")
            st.session_state.rag_pipeline = None
    
    # Initialize academic APIs and other components
    if 'academic_apis' not in st.session_state:
        st.session_state.academic_apis = AcademicAPIs()
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'selected_paper' not in st.session_state:
        st.session_state.selected_paper = None
    if 'reference_analyzer' not in st.session_state:
        st.session_state.reference_analyzer = ReferenceAnalyzer(st.session_state.academic_apis)
    if 'top_references' not in st.session_state:
        st.session_state.top_references = []
    if 'citation_network' not in st.session_state:
        st.session_state.citation_network = CitationNetwork()
    if 'batch_manager' not in st.session_state:
        st.session_state.batch_manager = BatchAnalysisManager(st.session_state.config)
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Upload PDF"
    if 'figures' not in st.session_state:
        st.session_state.figures = []
    if 'tables' not in st.session_state:
        st.session_state.tables = []
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = ""

def has_valid_api_key(config: ModelConfig) -> bool:
    """Check if there's a valid API key for the selected provider."""
    if config.model_provider == "openai":
        return bool(config.openai_api_key)
    elif config.model_provider == "anthropic":
        return bool(config.anthropic_api_key)
    elif config.model_provider == "google":
        return bool(config.google_api_key)
    return False

def reinit_rag_pipeline():
    """Reinitialize the RAG pipeline with current settings"""
    if has_valid_api_key(st.session_state.config):
        try:
            RAGPipeline = get_rag_pipeline()
            st.session_state.rag_pipeline = RAGPipeline(st.session_state.config)
            return True
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")
            st.session_state.rag_pipeline = None
            return False
    return False