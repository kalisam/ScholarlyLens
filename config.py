from dataclasses import dataclass

@dataclass
class ModelConfig:
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    llm_model: str = "deepseek-r1"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 2048
    search_k: int = 3

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