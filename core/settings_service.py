"""
Settings management service for ScholarLens
"""
import streamlit as st
from core.session import has_valid_api_key, reinit_rag_pipeline
from improvements.caching import setup_redis_cache

def update_api_keys(openai_key, anthropic_key, google_key):
    """
    Update API keys in the configuration.
    
    Args:
        openai_key: OpenAI API key
        anthropic_key: Anthropic API key
        google_key: Google API key
        
    Returns:
        Boolean indicating success of reinitialization
    """
    st.session_state.config.openai_api_key = openai_key
    st.session_state.config.anthropic_api_key = anthropic_key
    st.session_state.config.google_api_key = google_key
    
    # Reinitialize RAG pipeline with new keys
    if has_valid_api_key(st.session_state.config):
        return reinit_rag_pipeline()
    
    return False

def update_model_settings(model_provider, llm_model, embedding_model, temperature, max_tokens):
    """
    Update model settings in the configuration.
    
    Args:
        model_provider: Model provider name
        llm_model: Language model name
        embedding_model: Embedding model name
        temperature: Model temperature
        max_tokens: Max tokens for generation
        
    Returns:
        Boolean indicating success of reinitialization
    """
    needs_reinit = (
        model_provider != st.session_state.config.model_provider or
        llm_model != st.session_state.config.llm_model or
        embedding_model != st.session_state.config.embedding_model
    )
    
    # Update config
    st.session_state.config.model_provider = model_provider
    st.session_state.config.llm_model = llm_model
    st.session_state.config.embedding_model = embedding_model
    st.session_state.config.temperature = temperature
    st.session_state.config.max_tokens = max_tokens
    
    # Reinitialize RAG pipeline if needed
    if needs_reinit and has_valid_api_key(st.session_state.config):
        return reinit_rag_pipeline()
        
    return False

def update_processing_settings(chunk_size, chunk_overlap, search_k, extract_images, extract_tables, batch_workers):
    """
    Update PDF processing settings in the configuration.
    
    Args:
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        search_k: Number of search results
        extract_images: Whether to extract images
        extract_tables: Whether to extract tables
        batch_workers: Number of batch processing workers
    """
    st.session_state.config.chunk_size = chunk_size
    st.session_state.config.chunk_overlap = chunk_overlap
    st.session_state.config.search_k = search_k
    st.session_state.config.extract_images = extract_images
    st.session_state.config.extract_tables = extract_tables
    st.session_state.config.batch_workers = batch_workers

def update_cache_settings(use_cache, cache_ttl, use_redis, redis_host, redis_port, redis_password):
    """
    Update cache settings in the configuration.
    
    Args:
        use_cache: Whether to enable caching
        cache_ttl: Cache time-to-live in seconds
        use_redis: Whether to use Redis cache
        redis_host: Redis host
        redis_port: Redis port
        redis_password: Redis password
        
    Returns:
        Boolean indicating success of Redis connection
    """
    st.session_state.config.use_cache = use_cache
    st.session_state.config.cache_ttl = cache_ttl
    st.session_state.config.use_redis = use_redis
    st.session_state.config.redis_host = redis_host
    st.session_state.config.redis_port = redis_port
    st.session_state.config.redis_password = redis_password
    
    # Setup Redis cache if enabled
    if use_cache and use_redis:
        return setup_redis_cache(
            host=redis_host,
            port=redis_port,
            password=redis_password
        )
    
    return True

def clear_cache():
    """
    Clear all cached data.
    
    Returns:
        Boolean indicating success
    """
    if hasattr(st.session_state, 'rag_pipeline'):
        return st.session_state.rag_pipeline.clear_cache()
    return False