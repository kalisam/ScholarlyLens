from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import RetrievalQA
from typing import List, Optional
from langchain.schema import Document
from config import ModelConfig, PromptTemplates
import os

class RAGPipeline:
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Get API key from config or environment variables
        api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Please provide it via the UI or set the OPENAI_API_KEY environment variable.")
        
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=api_key
        )
        
        self.llm = ChatOpenAI(
            model_name=config.llm_model,
            openai_api_key=api_key,
            max_tokens=config.max_tokens
        )
        
        self.vector_store = None
        self.prompt_templates = PromptTemplates()

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create and store vectors from documents."""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        return self.vector_store

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
        novelty_prompt = PromptTemplate.from_template(
            self.prompt_templates.novelty_template
        )
        chain = LLMChain(llm=self.llm, prompt=novelty_prompt)
        return chain.run(text=text)

    def identify_research_gaps(self, text: str) -> str:
        """Identify potential research gaps."""
        gaps_prompt = PromptTemplate.from_template(
            self.prompt_templates.gaps_template
        )
        chain = LLMChain(llm=self.llm, prompt=gaps_prompt)
        return chain.run(text=text)