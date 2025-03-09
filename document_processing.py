from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from typing import Dict, List, Optional, Tuple, Any
from langchain.schema import Document
import spacy
from collections import defaultdict
import tempfile
import os
import fitz  # PyMuPDF
import base64
import io
from PIL import Image
import numpy as np
from config import ModelConfig

class ExtractedFigure:
    """Class representing an extracted figure from a PDF."""
    def __init__(self, image_data: bytes, caption: str, page_num: int):
        self.image_data = image_data
        self.caption = caption
        self.page_num = page_num
        
    def get_base64_image(self) -> str:
        """Get the image as a base64 encoded string for display."""
        return base64.b64encode(self.image_data).decode('utf-8')

class ExtractedTable:
    """Class representing an extracted table from a PDF."""
    def __init__(self, data: List[List[str]], caption: str, page_num: int):
        self.data = data
        self.caption = caption
        self.page_num = page_num

class DocumentProcessor:
    def __init__(self, config: ModelConfig):
        self.config = config
        # Initialize spaCy with standard English model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def load_pdf(self, file_content: bytes) -> List[Document]:
        """Load PDF content and return LangChain documents."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_pdf_path = temp_file.name
            loader = PDFPlumberLoader(temp_pdf_path)
            return loader.load()

    def load_pdf_with_pymupdf(self, file_content: bytes) -> Tuple[List[Document], List[ExtractedFigure], List[ExtractedTable]]:
        """
        Load PDF content using PyMuPDF for improved extraction including figures and tables.
        
        Args:
            file_content: PDF file content as bytes
            
        Returns:
            Tuple of (text documents, figures, tables)
        """
        documents = []
        figures = []
        tables = []
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_pdf_path = temp_file.name
            
            try:
                # Open the PDF with PyMuPDF
                pdf_doc = fitz.open(temp_pdf_path)
                
                # Extract text and media elements
                for page_num, page in enumerate(pdf_doc):
                    # Extract text
                    text = page.get_text()
                    documents.append(Document(
                        page_content=text,
                        metadata={"page": page_num, "source": "pymupdf"}
                    ))
                    
                    # Extract images
                    image_list = page.get_images(full=True)
                    for img_idx, img in enumerate(image_list):
                        xref = img[0]
                        base_image = pdf_doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Try to find caption (simple heuristic - look for "Figure" or "Fig." near the image)
                        caption = self._find_caption_for_image(text, img_idx)
                        
                        figures.append(ExtractedFigure(
                            image_data=image_bytes,
                            caption=caption,
                            page_num=page_num
                        ))
                    
                    # Extract tables (simplified approach - looks for tabular content)
                    tables_on_page = self._extract_tables_from_page(page)
                    for table_idx, table_data in enumerate(tables_on_page):
                        caption = self._find_caption_for_table(text, table_idx)
                        tables.append(ExtractedTable(
                            data=table_data,
                            caption=caption,
                            page_num=page_num
                        ))
                
                pdf_doc.close()
            finally:
                os.unlink(temp_pdf_path)
                
        return documents, figures, tables
    
    def _find_caption_for_image(self, text: str, img_idx: int) -> str:
        """Find a likely caption for an image based on surrounding text."""
        # Look for patterns like "Figure X:" or "Fig. X:"
        import re
        patterns = [
            rf"(?:Figure|Fig\.)\s*{img_idx+1}\s*[:\.]\s*([^\n]+)",
            rf"(?:Figure|Fig\.)\s*{img_idx+1}\s*[\.\-]\s*([^\n]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return f"Figure {img_idx+1}"
    
    def _find_caption_for_table(self, text: str, table_idx: int) -> str:
        """Find a likely caption for a table based on surrounding text."""
        import re
        patterns = [
            rf"(?:Table)\s*{table_idx+1}\s*[:\.]\s*([^\n]+)",
            rf"(?:Table)\s*{table_idx+1}\s*[\.\-]\s*([^\n]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return f"Table {table_idx+1}"
    
    def _extract_tables_from_page(self, page) -> List[List[List[str]]]:
        """Extract tables from a PDF page using PyMuPDF."""
        tables = []
        
        # PyMuPDF doesn't have direct table extraction, so this is a basic approach
        # In a full implementation, this would use more advanced table detection
        
        # Check if the page has tabular data
        text_blocks = page.get_text("blocks")
        for block in text_blocks:
            text = block[4]
            # Simple heuristic: if text contains multiple tab characters or has consistent spacing
            # it might be a table
            if '\t' in text or self._looks_like_table(text):
                rows = text.split('\n')
                table_data = []
                for row in rows:
                    if '\t' in row:
                        cells = row.split('\t')
                    else:
                        # Try to split based on consistent spacing
                        cells = self._split_table_row(row)
                    if len(cells) > 1:  # Only include if it looks like a row
                        table_data.append(cells)
                
                if len(table_data) > 1:  # Only include if it has multiple rows
                    tables.append(table_data)
        
        return tables
    
    def _looks_like_table(self, text: str) -> bool:
        """Check if text looks like it might be a table based on structure."""
        lines = text.split('\n')
        if len(lines) < 2:
            return False
        
        # Check if lines have consistent number of whitespace-separated elements
        counts = [len(line.split()) for line in lines if line.strip()]
        if len(set(counts)) <= 2 and max(counts, default=0) > 2:
            return True
        
        return False
    
    def _split_table_row(self, row: str) -> List[str]:
        """Split a table row based on whitespace patterns."""
        import re
        # Look for groups of whitespace that might separate columns
        return [cell.strip() for cell in re.split(r'\s{2,}', row) if cell.strip()]

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
    
    def extract_sections_improved(self, documents: List[Document]) -> Dict[str, str]:
        """
        Improved section extraction with better pattern matching and handling for 
        various section naming conventions.
        """
        full_text = " ".join([doc.page_content for doc in documents])
        
        # Define section mapping with various possible headings
        section_patterns = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'background', 'overview'],
            'methods': ['method', 'methodology', 'materials and methods', 'experimental setup', 'approach'],
            'results': ['result', 'findings', 'outcomes', 'observations'],
            'discussion': ['discussion', 'analysis', 'interpretation'],
            'conclusion': ['conclusion', 'concluding remarks', 'summary and conclusion', 'future work']
        }
        
        # Initialize sections
        sections = {key: '' for key in section_patterns.keys()}
        
        # Process text for better section detection
        current_section = None
        lines = full_text.split('\n')
        line_idx = 0
        
        while line_idx < len(lines):
            line = lines[line_idx]
            line_lower = line.lower()
            
            # Check if this line indicates a section header
            new_section = None
            for section, patterns in section_patterns.items():
                for pattern in patterns:
                    # Check various patterns: standalone heading, numbered heading, etc.
                    if (re.search(r'^' + pattern + r'[ :]*$', line_lower) or 
                        re.search(r'^[\d\.]*\s*' + pattern + r'[ :]*$', line_lower) or
                        re.search(r'^[\d\.]*\s*' + pattern + r'[ :\.]', line_lower)):
                        new_section = section
                        break
                if new_section:
                    break
            
            if new_section:
                current_section = new_section
                # Skip the header line
                line_idx += 1
            elif current_section:
                sections[current_section] += line + '\n'
                line_idx += 1
            else:
                # If we haven't identified a section yet but have some text,
                # it might be the abstract (often appears before any heading)
                if line.strip() and not current_section and sections['abstract'] == '':
                    sections['abstract'] += line + '\n'
                line_idx += 1
        
        return sections

    def identify_key_concepts(self, text: str) -> Dict[str, set]:
        """Extract key concepts using NER."""
        doc = self.nlp(text)
        concepts = defaultdict(set)
        
        # Include relevant entity types from standard model
        relevant_types = {'ORG', 'PERSON', 'GPE', 'WORK_OF_ART', 'DATE', 'NORP'}
        
        for ent in doc.ents:
            if ent.label_ in relevant_types:
                concepts[ent.label_].add(ent.text)
            
        return dict(concepts)
    
    def identify_key_concepts_improved(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Enhanced key concept extraction with frequency, context, and relations.
        
        Returns a dictionary with entity types as keys and lists of entity information as values.
        Each entity information includes the text, frequency, and sample contexts.
        """
        # Process with spaCy
        doc = self.nlp(text)
        
        # Track entities and their occurrences
        entity_occurrences = defaultdict(list)
        
        # Track all entities for frequency counting
        all_entities = []
        
        for ent in doc.ents:
            all_entities.append((ent.text, ent.label_))
            
            # Get context (text around the entity)
            start_idx = max(0, ent.start - 5)
            end_idx = min(len(doc), ent.end + 5)
            context = doc[start_idx:end_idx].text
            
            entity_occurrences[ent.label_].append({
                'text': ent.text,
                'context': context
            })
        
        # Count frequencies
        entity_counter = defaultdict(lambda: defaultdict(int))
        for text, label in all_entities:
            entity_counter[label][text] += 1
        
        # Build final result
        result = {}
        for entity_type, entities in entity_occurrences.items():
            # Group by entity text
            grouped_entities = defaultdict(list)
            for entity in entities:
                grouped_entities[entity['text']].append(entity['context'])
            
            # Create final entries
            result[entity_type] = [
                {
                    'text': entity_text,
                    'frequency': entity_counter[entity_type][entity_text],
                    'contexts': contexts[:3]  # Limit to 3 sample contexts
                }
                for entity_text, contexts in grouped_entities.items()
            ]
            
            # Sort by frequency
            result[entity_type] = sorted(
                result[entity_type], 
                key=lambda x: x['frequency'], 
                reverse=True
            )
        
        return result
    
    def extract_keywords_and_phrases(self, text: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Extract important keywords and phrases using textrank-like algorithm.
        
        Args:
            text: The text to analyze
            top_n: Number of top keywords to return
            
        Returns:
            List of dictionaries with keywords and their scores
        """
        # Process with spaCy
        doc = self.nlp(text)
        
        # Filter for relevant tokens
        keywords = []
        for token in doc:
            if (token.is_alpha and not token.is_stop and not token.is_punct and 
                token.pos_ in ['NOUN', 'ADJ', 'VERB', 'PROPN']):
                keywords.append({
                    'text': token.text,
                    'lemma': token.lemma_,
                    'pos': token.pos_
                })
        
        # Count frequencies
        keyword_counter = defaultdict(int)
        for keyword in keywords:
            keyword_counter[keyword['lemma']] += 1
        
        # Get phrases (simplified implementation - in production, use a proper keyphrase extraction)
        phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk) > 1 and not any(token.is_stop for token in chunk):
                phrase_text = chunk.text
                phrases.append({
                    'text': phrase_text,
                    'length': len(chunk)
                })
        
        # Count phrase frequencies
        phrase_counter = defaultdict(int)
        for phrase in phrases:
            phrase_counter[phrase['text']] += 1
        
        # Combine keywords and phrases with scores
        combined_results = []
        
        # Add top keywords
        for lemma, count in sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            # Find the most common form of this lemma
            forms = [k['text'] for k in keywords if k['lemma'] == lemma]
            most_common_form = max(set(forms), key=forms.count)
            
            combined_results.append({
                'text': most_common_form,
                'type': 'keyword',
                'count': count,
                'score': count / len(keywords) if keywords else 0
            })
        
        # Add top phrases
        for phrase, count in sorted(phrase_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            combined_results.append({
                'text': phrase,
                'type': 'phrase',
                'count': count,
                'score': count / len(phrases) if phrases else 0
            })
        
        # Sort by score and return top N
        return sorted(combined_results, key=lambda x: x['score'], reverse=True)[:top_n]