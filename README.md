# Research Paper Analysis Tool

An advanced tool for analyzing research papers using RAG (Retrieval Augmented Generation) and LLMs. This tool helps researchers identify research gaps, assess novelty, extract key concepts, and interact with academic papers through natural language queries.

## Features

- **PDF Processing**: Automatically extracts and processes PDF research papers
- **Section Analysis**: Identifies and separates key paper sections (abstract, introduction, methods, etc.)
- **Key Concept Extraction**: Uses scientific NER to identify important concepts and entities
- **Research Gap Analysis**: Identifies potential research gaps and future work opportunities
- **Novelty Assessment**: Analyzes the paper's novel contributions and approaches
- **Interactive Q&A**: Allows natural language queries about the paper's content

## Project Structure

```
deepseek_chatbot/
│── app.py              # Streamlit UI implementation
│── document_processing.py  # PDF extraction & processing
│── rag_functions.py    # Vector store and LLM pipeline
│── config.py           # Configuration settings
│── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deepseek_chatbot
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the scientific NLP model:
```bash
python -m spacy download en_core_sci_sm
```

5. Install and start Ollama:
- Follow instructions at [Ollama's website](https://ollama.ai)
- Pull the DeepSeek model:
```bash
ollama pull deepseek-r1
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload a research paper PDF

4. Explore different analyses through the tabs:
   - Paper Structure: View different sections of the paper
   - Key Concepts: Explore important terms and concepts
   - Research Gaps: Identify potential areas for future research
   - Novelty Analysis: Understand the paper's unique contributions

5. Use the Q&A interface to ask specific questions about the paper

## Requirements

- Python 3.8+
- Ollama
- 8GB+ RAM recommended
- GPU optional but recommended for better performance

## Technical Details

### Components

- **DocumentProcessor**: Handles PDF processing and text extraction
  - Uses PDFPlumber for text extraction
  - Implements semantic chunking for better context preservation
  - Extracts paper sections using pattern recognition

- **RAGPipeline**: Manages the RAG implementation
  - Uses FAISS for vector storage
  - Implements HuggingFace embeddings
  - Manages LLM interaction through Ollama

- **Configuration**: Centralized settings management
  - Model parameters
  - Chunking settings
  - Prompt templates

### Models Used

- Embeddings: sentence-transformers/all-mpnet-base-v2
- LLM: DeepSeek through Ollama
- NER: SpaCy's en_core_sci_sm model

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgments

- LangChain for the RAG implementation
- Streamlit for the UI framework
- HuggingFace for embeddings
- Ollama for LLM access
- SpaCy for NER capabilities

## Roadmap

- [ ] Integration with academic APIs (arXiv, Semantic Scholar)
- [ ] Enhanced citation analysis
- [ ] Batch processing capabilities
- [ ] Export functionality for analyses
- [ ] Visualization improvements