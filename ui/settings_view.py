"""
Settings view for ScholarLens
"""
import streamlit as st
from core.settings_service import (
    update_api_keys,
    update_model_settings,
    update_processing_settings,
    update_cache_settings,
    clear_cache
)

def render_settings_view():
    """Render the settings interface"""
    st.header("Settings")
    
    # Create tabs for different settings
    tab1, tab2, tab3, tab4 = st.tabs([
        "API Keys", 
        "Model Settings",
        "Processing Settings",
        "Cache Settings"
    ])
    
    with tab1:
        render_api_keys_tab()
    
    with tab2:
        render_model_settings_tab()
    
    with tab3:
        render_processing_settings_tab()
    
    with tab4:
        render_cache_settings_tab()

def render_api_keys_tab():
    """Render API Keys settings tab"""
    st.subheader("API Keys")
    
    # OpenAI API key
    openai_key = st.text_input(
        "OpenAI API Key", 
        value=st.session_state.config.openai_api_key,
        type="password"
    )
    
    # Anthropic API key
    anthropic_key = st.text_input(
        "Anthropic API Key", 
        value=st.session_state.config.anthropic_api_key,
        type="password"
    )
    
    # Google API key
    google_key = st.text_input(
        "Google API Key", 
        value=st.session_state.config.google_api_key,
        type="password"
    )
    
    # Update API keys
    if st.button("Save API Keys"):
        success = update_api_keys(openai_key, anthropic_key, google_key)
        if success:
            st.success("API keys updated and RAG pipeline reinitialized!")
        else:
            st.warning("API keys updated, but no valid API key provided for the selected model provider.")
        
def render_model_settings_tab():
    """Render Model Settings tab"""
    st.subheader("Model Settings")
    
    # Model provider
    model_provider = st.selectbox(
        "Model Provider",
        ["openai", "anthropic", "google"],
        index=["openai", "anthropic", "google"].index(st.session_state.config.model_provider)
    )
    
    # Available models based on provider
    available_models = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "google": ["gemini-pro", "gemini-1.5-pro"]
    }.get(model_provider, ["gpt-4o"])
    
    # Default model
    default_model = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-haiku-20240307",
        "google": "gemini-pro"
    }.get(model_provider, "gpt-4o")
    
    # Current model or default
    current_model = st.session_state.config.llm_model
    if current_model not in available_models:
        current_model = default_model
    
    # LLM model
    llm_model = st.selectbox(
        "Language Model",
        available_models,
        index=available_models.index(current_model) if current_model in available_models else 0
    )
    
    # Embedding model (only for OpenAI)
    embedding_model = st.selectbox(
        "Embedding Model",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=["text-embedding-3-small", "text-embedding-3-large"].index(
            st.session_state.config.embedding_model
        ) if st.session_state.config.embedding_model in ["text-embedding-3-small", "text-embedding-3-large"] else 0
    )
    
    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.config.temperature,
        step=0.1
    )
    
    # Max tokens
    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=4096,
        value=st.session_state.config.max_tokens,
        step=256
    )
    
    # Update model settings
    if st.button("Save Model Settings"):
        success = update_model_settings(
            model_provider, llm_model, embedding_model, temperature, max_tokens
        )
        if success:
            st.success("Model settings updated and RAG pipeline reinitialized!")
        else:
            st.success("Model settings updated!")
            
def render_processing_settings_tab():
    """Render Processing Settings tab"""
    st.subheader("Processing Settings")
    
    # Chunk size
    chunk_size = st.slider(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=st.session_state.config.chunk_size,
        step=100
    )
    
    # Chunk overlap
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=st.session_state.config.chunk_overlap,
        step=50
    )
    
    # Search k
    search_k = st.slider(
        "Number of search results (k)",
        min_value=1,
        max_value=10,
        value=st.session_state.config.search_k,
        step=1
    )
    
    # Extract images and tables
    extract_images = st.checkbox(
        "Extract images from PDFs",
        value=st.session_state.config.extract_images
    )
    
    extract_tables = st.checkbox(
        "Extract tables from PDFs",
        value=st.session_state.config.extract_tables
    )
    
    # Batch workers
    batch_workers = st.slider(
        "Batch processing workers",
        min_value=1,
        max_value=8,
        value=st.session_state.config.batch_workers,
        step=1
    )
    
    # Update processing settings
    if st.button("Save Processing Settings"):
        update_processing_settings(
            chunk_size, chunk_overlap, search_k, 
            extract_images, extract_tables, batch_workers
        )
        st.success("Processing settings updated!")
            
def render_cache_settings_tab():
    """Render Cache Settings tab"""
    st.subheader("Cache Settings")
    
    # Use cache
    use_cache = st.checkbox(
        "Enable caching",
        value=st.session_state.config.use_cache
    )
    
    # Cache TTL
    cache_ttl = st.slider(
        "Cache TTL (seconds)",
        min_value=3600,
        max_value=604800,  # 1 week
        value=st.session_state.config.cache_ttl,
        step=3600
    )
    
    # Use Redis
    use_redis = st.checkbox(
        "Use Redis cache (instead of file cache)",
        value=st.session_state.config.use_redis
    )
    
    # Redis settings
    redis_host = st.text_input(
        "Redis Host",
        value=st.session_state.config.redis_host
    )
    
    redis_port = st.number_input(
        "Redis Port",
        value=st.session_state.config.redis_port,
        min_value=1,
        max_value=65535
    )
    
    redis_password = st.text_input(
        "Redis Password",
        value=st.session_state.config.redis_password,
        type="password"
    )
    
    # Update cache settings
    if st.button("Save Cache Settings"):
        success = update_cache_settings(
            use_cache, cache_ttl, use_redis,
            redis_host, redis_port, redis_password
        )
        if success:
            st.success("Cache settings updated and Redis cache connected!")
        else:
            st.warning("Cache settings updated but Redis connection failed. Using file cache.")
    
    # Clear cache button
    if st.button("Clear Cache"):
        success = clear_cache()
        if success:
            st.success("Cache cleared successfully!")
        else:
            st.warning("Cache is not enabled or could not be cleared.")