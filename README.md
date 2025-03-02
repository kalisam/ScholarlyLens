# ScholarLens: Paper Analysis Chatbot

An advanced tool for analyzing research papers using RAG (Retrieval Augmented Generation) and OpenAI's language models. This tool helps researchers identify research gaps, assess novelty, extract key concepts, and interact with academic papers through natural language queries.

## Features

- **PDF Processing**: Automatically extracts and processes PDF research papers
- **Academic API Integration**: Search and analyze papers directly from arXiv and Semantic Scholar
- **Section Analysis**: Identifies and separates key paper sections (abstract, introduction, methods, etc.)
- **Key Concept Extraction**: Uses scientific NER to identify important concepts and entities
- **Research Gap Analysis**: Identifies potential research gaps and future work opportunities
- **Novelty Assessment**: Analyzes the paper's novel contributions and approaches
- **Interactive Q&A**: Allows natural language queries about the paper's content

## Project Structure

```
ScholarLens/
│── app.py              # Streamlit UI implementation
│── document_processing.py  # PDF extraction & processing
│── rag_functions.py    # Vector store and LLM pipeline
│── academic_apis.py    # Academic database API integrations
│── config.py           # Configuration settings
│── requirements.txt    # Project dependencies
│── .env.example        # Template for environment variables
│── .gitignore          # Files to exclude from version control
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/realjules/ScholarLens.git
cd ScholarLens
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# OR
source venv/bin/activate      # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the standard spaCy model:
```bash
python -m spacy download en_core_web_sm
```

5. Set up OpenAI API key:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. If you haven't set the API key in the `.env` file, you'll be prompted to enter it in the application

4. Upload a research paper PDF

5. Explore different analyses through the tabs:
   - Paper Structure: View different sections of the paper
   - Key Concepts: Explore important terms and concepts
   - Research Gaps: Identify potential areas for future research
   - Novelty Analysis: Understand the paper's unique contributions

6. Alternatively, use the "Search Academic Databases" tab to find papers on arXiv or Semantic Scholar

7. Use the Q&A interface to ask specific questions about the paper

## Requirements

- Python 3.8+
- OpenAI API key
- 8GB+ RAM recommended

## Technical Details

### Components

- **DocumentProcessor**: Handles PDF processing and text extraction
  - Uses PDFPlumber for text extraction
  - Implements semantic chunking for better context preservation
  - Extracts paper sections using pattern recognition

- **RAGPipeline**: Manages the RAG implementation
  - Uses FAISS for vector storage
  - Implements OpenAI embeddings
  - Manages LLM interaction through OpenAI API

- **Configuration**: Centralized settings management
  - Model parameters
  - Chunking settings
  - Prompt templates

### Models Used

- Embeddings: OpenAI text-embedding-3-small
- LLM: OpenAI GPT-4o
- NER: SpaCy's en_core_web_sm model

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- LangChain for the RAG implementation
- Streamlit for the UI framework
- OpenAI for language model and embeddings
- SpaCy for NER capabilities

## Roadmap

- [x] Integration with academic APIs (arXiv, Semantic Scholar)
- [ ] Enhanced citation analysis
- [ ] Batch processing capabilities
- [ ] Export functionality for analyses
- [ ] Visualization improvements
- [ ] Citation network analysis
- [ ] Research trend identification