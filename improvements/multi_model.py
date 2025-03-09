from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from langchain.schema import Document
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import ModelConfig


class ModelProvider(ABC):
    """Abstract base class for different model providers."""
    
    @abstractmethod
    def get_llm(self, model_config: ModelConfig):
        """Get the language model."""
        pass
    
    @abstractmethod
    def get_embeddings(self, model_config: ModelConfig):
        """Get the embeddings model."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Get list of available models."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Get the default model."""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI model provider."""
    
    def get_llm(self, model_config: ModelConfig):
        """Get the OpenAI language model."""
        return ChatOpenAI(
            model_name=model_config.llm_model,
            openai_api_key=model_config.openai_api_key,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature
        )
    
    def get_embeddings(self, model_config: ModelConfig):
        """Get the OpenAI embeddings model."""
        return OpenAIEmbeddings(
            model=model_config.embedding_model,
            openai_api_key=model_config.openai_api_key
        )
    
    @property
    def provider_name(self) -> str:
        return "OpenAI"
    
    @property
    def available_models(self) -> List[str]:
        return ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    
    @property
    def default_model(self) -> str:
        return "gpt-4o"
    
    @property
    def available_embedding_models(self) -> List[str]:
        return ["text-embedding-3-small", "text-embedding-3-large"]
    
    @property
    def default_embedding_model(self) -> str:
        return "text-embedding-3-small"


class AnthropicProvider(ModelProvider):
    """Anthropic model provider."""
    
    def get_llm(self, model_config: ModelConfig):
        """Get the Anthropic language model."""
        return ChatAnthropic(
            model_name=model_config.llm_model or "claude-3-haiku-20240307",
            anthropic_api_key=model_config.anthropic_api_key,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature
        )
    
    def get_embeddings(self, model_config: ModelConfig):
        """
        Get embeddings - Anthropic doesn't have embeddings yet,
        so we use OpenAI's or HuggingFace embeddings as fallback
        """
        if model_config.openai_api_key:
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=model_config.openai_api_key
            )
        else:
            # Fallback to local HuggingFace embeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
    
    @property
    def provider_name(self) -> str:
        return "Anthropic"
    
    @property
    def available_models(self) -> List[str]:
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    
    @property
    def default_model(self) -> str:
        return "claude-3-haiku-20240307"


class GoogleProvider(ModelProvider):
    """Google AI model provider."""
    
    def get_llm(self, model_config: ModelConfig):
        """Get the Google language model."""
        return ChatGoogleGenerativeAI(
            model=model_config.llm_model or "gemini-pro",
            google_api_key=model_config.google_api_key,
            temperature=model_config.temperature,
            max_output_tokens=model_config.max_tokens
        )
    
    def get_embeddings(self, model_config: ModelConfig):
        """Get the Google embeddings model."""
        if hasattr(model_config, 'use_google_embeddings') and model_config.use_google_embeddings:
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=model_config.google_api_key
            )
        else:
            # Fallback to local HuggingFace embeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
    
    @property
    def provider_name(self) -> str:
        return "Google"
    
    @property
    def available_models(self) -> List[str]:
        return ["gemini-pro", "gemini-1.5-pro"]
    
    @property
    def default_model(self) -> str:
        return "gemini-pro"


class ModelFactory:
    """Factory class to create model providers."""
    
    @staticmethod
    def get_provider(provider_name: str) -> ModelProvider:
        """
        Get a model provider by name.
        
        Args:
            provider_name: Name of the provider ("openai", "anthropic", "google")
            
        Returns:
            ModelProvider instance
            
        Raises:
            ValueError: If provider_name is not supported
        """
        providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "google": GoogleProvider()
        }
        
        provider = providers.get(provider_name.lower())
        if not provider:
            raise ValueError(f"Unsupported model provider: {provider_name}")
        
        return provider
    
    @staticmethod
    def get_all_providers() -> Dict[str, ModelProvider]:
        """
        Get all available model providers.
        
        Returns:
            Dictionary of provider name to ModelProvider
        """
        return {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "google": GoogleProvider()
        }
    
    @staticmethod
    def check_api_key(provider_name: str, model_config: ModelConfig) -> bool:
        """
        Check if the API key for a provider is set in model_config.
        
        Args:
            provider_name: Name of the provider
            model_config: ModelConfig instance
            
        Returns:
            True if the API key is set, False otherwise
        """
        if provider_name.lower() == "openai":
            return bool(model_config.openai_api_key)
        elif provider_name.lower() == "anthropic":
            return bool(model_config.anthropic_api_key)
        elif provider_name.lower() == "google":
            return bool(model_config.google_api_key)
        else:
            return False