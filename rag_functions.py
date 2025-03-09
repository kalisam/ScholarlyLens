from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from typing import List, Optional, Dict, Any, Union
from langchain.schema import Document
from config import ModelConfig, PromptTemplates
import os
import logging
import hashlib
import time
import json
import sys

# Add the parent directory to sys.path to allow direct imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
# Direct imports to avoid circular dependencies
import improvements.caching
from improvements.multi_model import ModelFactory

# Get cache manager from caching module
cache_manager = improvements.caching.cache_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Get the model provider
        try:
            self.model_provider = ModelFactory.get_provider(config.model_provider)
        except ValueError:
            logger.warning(f"Unknown model provider: {config.model_provider}, falling back to OpenAI")
            self.model_provider = ModelFactory.get_provider("openai")
            
        # Setup embeddings based on provider
        try:
            self.embeddings = self.model_provider.get_embeddings(config)
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            # Fallback to OpenAI embeddings if available
            if config.openai_api_key:
                logger.info("Falling back to OpenAI embeddings")
                self.embeddings = OpenAIEmbeddings(
                    model=config.embedding_model,
                    openai_api_key=config.openai_api_key
                )
            else:
                raise ValueError("Could not initialize embeddings. Please check your API keys.")
        
        # Setup LLM based on provider
        try:
            self.llm = self.model_provider.get_llm(config)
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise ValueError(f"Could not initialize LLM: {e}")
        
        self.vector_store = None
        self.prompt_templates = PromptTemplates()
        
        # Setup caching if enabled
        if config.use_cache:
            if config.use_redis:
                from improvements.caching import setup_redis_cache
                redis_success = setup_redis_cache(
                    host=config.redis_host,
                    port=config.redis_port,
                    password=config.redis_password
                )
                if redis_success:
                    logger.info("Redis cache enabled")
                else:
                    logger.info("Redis cache setup failed, using file cache")
            else:
                logger.info("File cache enabled")

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create and store vectors from documents."""
        # Generate a unique key for this set of documents
        doc_hash = self._hash_documents(documents)
        cache_key = f"vector_store:{doc_hash}"
        
        # Check if we have a cached vector store
        if self.config.use_cache and cache_manager.has(cache_key):
            logger.info("Using cached vector store")
            self.vector_store = cache_manager.get(cache_key)
            if self.vector_store:
                return self.vector_store
        
        # Create a new vector store
        logger.info("Creating new vector store")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Cache the vector store if caching is enabled
        if self.config.use_cache:
            logger.debug(f"Caching vector store with key {cache_key}")
            cache_manager.set(
                cache_key, 
                self.vector_store, 
                expiration=self.config.cache_ttl
            )
        
        return self.vector_store
    
    def _hash_documents(self, documents: List[Document]) -> str:
        """Create a hash of document contents to use as a cache key."""
        content = "".join([doc.page_content for doc in documents])
        return hashlib.md5(content.encode()).hexdigest()

    def get_retriever(self):
        """Get retriever from vector store."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please create it first.")
        return self.vector_store.as_retriever(
            search_kwargs={"k": self.config.search_k}
        )

    def create_qa_chain(self) -> RetrievalQA:
        """Create QA chain for document interaction."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please create it first.")
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.get_retriever()
        )

    def analyze_novelty(self, text: str) -> str:
        """Analyze the novelty of the research paper."""
        logger.info("Analyzing novelty")
        novelty_prompt = PromptTemplate.from_template(
            self.prompt_templates.novelty_template
        )
        chain = LLMChain(llm=self.llm, prompt=novelty_prompt)
        return chain.run(text=text)

    def identify_research_gaps(self, text: str) -> str:
        """Identify potential research gaps."""
        logger.info("Identifying research gaps")
        gaps_prompt = PromptTemplate.from_template(
            self.prompt_templates.gaps_template
        )
        chain = LLMChain(llm=self.llm, prompt=gaps_prompt)
        return chain.run(text=text)
        
    def generate_paper_summary(self, text: str) -> str:
        """Generate a concise summary of the paper."""
        logger.info("Generating paper summary")
        summary_prompt = PromptTemplate.from_template(
            self.prompt_templates.paper_summary_template
        )
        chain = LLMChain(llm=self.llm, prompt=summary_prompt)
        return chain.run(text=text)
    
    def generate_paper_recommendations(self, text: str) -> str:
        """Generate recommendations for related papers."""
        logger.info("Generating paper recommendations")
        recommendations_prompt = PromptTemplate.from_template(
            self.prompt_templates.paper_recommendation_template
        )
        chain = LLMChain(llm=self.llm, prompt=recommendations_prompt)
        return chain.run(text=text)
    
    def run_query(self, query: str) -> str:
        """
        Run a query against the RAG pipeline with caching.
        
        Args:
            query: The query string
            
        Returns:
            The response from the model
        """
        # Generate a cache key
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Please create it first.")
            
        cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
        
        # Check if we have a cached result
        if self.config.use_cache and cache_manager.has(cache_key):
            logger.info(f"Using cached query result for: {query}")
            return cache_manager.get(cache_key)
        
        # Create and run the QA chain
        qa_chain = self.create_qa_chain()
        result = qa_chain.run(query)
        
        # Cache the result
        if self.config.use_cache:
            logger.debug(f"Caching query result for: {query}")
            cache_manager.set(cache_key, result, expiration=self.config.cache_ttl)
        
        return result
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.config.use_cache:
            logger.info("Clearing RAG pipeline cache")
            cache_manager.clear_all("vector_store:")
            cache_manager.clear_all("novelty:")
            cache_manager.clear_all("gaps:")
            cache_manager.clear_all("summary:")
            cache_manager.clear_all("recommendations:")
            cache_manager.clear_all("query:")
            return True
        return False