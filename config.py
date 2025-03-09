from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class ModelConfig:
    # Embedding model settings
    embedding_model: str = "text-embedding-3-small"
    
    # LLM model settings
    llm_model: str = "gpt-4o"
    model_provider: str = "openai"  # "openai", "anthropic", "google"
    temperature: float = 0.2
    max_tokens: int = 2048
    
    # Document processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    search_k: int = 3
    extract_images: bool = True
    extract_tables: bool = True
    
    # API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    
    # Cache settings
    use_cache: bool = True
    cache_ttl: int = 86400  # 24 hours in seconds
    use_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    
    # Advanced settings
    use_google_embeddings: bool = False
    
    # Batch processing settings
    batch_workers: int = 2
    batch_result_dir: str = "batch_results"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "model_provider": self.model_provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "search_k": self.search_k,
            "extract_images": self.extract_images,
            "extract_tables": self.extract_tables,
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl,
            "use_redis": self.use_redis,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "use_google_embeddings": self.use_google_embeddings,
            "batch_workers": self.batch_workers,
            "batch_result_dir": self.batch_result_dir
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        config = cls()
        # Update only fields that are present in the data
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

@dataclass
class PromptTemplates:
    qa_template: str = """  
        1. Use ONLY the context below.  
        2. If unsure, say "I don't know".  
        3. Keep answers under 4 sentences.  

        Context: {context}  
        Question: {question}  
        Answer:  
    """
    
    novelty_template: str = """
        Analyze the novelty of this research paper section. Focus on:
        1. New methods or approaches
        2. Novel datasets or resources
        3. Unique problem formulations
        4. Original findings or insights

        Text: {text}
        Novelty Analysis:
    """
    
    gaps_template: str = """
        Analyze this research paper section and identify potential research gaps. Consider:
        1. Unanswered questions
        2. Limitations mentioned
        3. Future work suggestions
        4. Contradictions or inconsistencies

        Text: {text}
        Research Gaps:
    """
    
    paper_recommendation_template: str = """
        Based on this paper's content, suggest related papers that the reader might find interesting.
        Focus on papers that:
        1. Extend or build upon the methods used
        2. Address similar research questions
        3. Offer alternative approaches to the same problem
        4. Were published recently (prefer more recent work)

        Text: {text}
        Related Paper Recommendations:
    """
    
    paper_summary_template: str = """
        Provide a concise summary of this research paper. Include:
        1. The main research question or objective
        2. The key methods used
        3. The most important findings
        4. The significance of the results

        Keep the summary to 3-5 sentences.

        Text: {text}
        Summary:
    """