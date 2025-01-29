from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from typing import Dict, List
from langchain.schema import Document
import spacy
from collections import defaultdict
import tempfile
from config import ModelConfig

class DocumentProcessor:
    def __init__(self, config: ModelConfig):
        self.config = config
        # Initialize spaCy for scientific NER
        try:
            self.nlp = spacy.load("en_core_sci_sm")
        except:
            import os
            os.system("python -m spacy download en_core_sci_sm")
            self.nlp = spacy.load("en_core_sci_sm")

    def load_pdf(self, file_content: bytes) -> List[Document]:
        """Load PDF content and return LangChain documents."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_pdf_path = temp_file.name
            loader = PDFPlumberLoader(temp_pdf_path)
            return loader.load()

    def create_chunks(self, documents: List[Document], embeddings) -> List[Document]:
        """Split documents into semantic chunks."""
        text_splitter = SemanticChunker(embeddings)
        return text_splitter.split_documents(documents)

    def extract_sections(self, documents: List[Document]) -> Dict[str, str]:
        """Extract different sections of the research paper."""
        full_text = " ".join([doc.page_content for doc in documents])
        
        sections = {
            'abstract': '',
            'introduction': '',
            'methods': '',
            'results': '',
            'discussion': '',
            'conclusion': ''
        }
        
        # Simple regex-based section extraction
        current_section = None
        lines = full_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            if 'abstract' in line_lower:
                current_section = 'abstract'
            elif 'introduction' in line_lower:
                current_section = 'introduction'
            elif 'method' in line_lower:
                current_section = 'methods'
            elif 'result' in line_lower:
                current_section = 'results'
            elif 'discussion' in line_lower:
                current_section = 'discussion'
            elif 'conclusion' in line_lower:
                current_section = 'conclusion'
            elif current_section:
                sections[current_section] += line + '\n'
                
        return sections

    def identify_key_concepts(self, text: str) -> Dict[str, set]:
        """Extract key scientific concepts using NER."""
        doc = self.nlp(text)
        concepts = defaultdict(set)
        
        for ent in doc.ents:
            concepts[ent.label_].add(ent.text)
            
        return dict(concepts)