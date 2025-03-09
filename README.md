# ScholarLens: Advanced Research Paper Analysis

A comprehensive tool for analyzing research papers using RAG (Retrieval Augmented Generation) with multiple AI models (OpenAI, Anthropic, Google). This tool helps researchers identify research gaps, assess novelty, extract key concepts, visualize citation networks, and interact with academic papers through natural language queries.

## Features

- **PDF Processing**: Extracts text, figures, tables and processes PDF research papers
- **Multi-model Support**: Works with OpenAI, Anthropic Claude, and Google Gemini models
- **Academic API Integration**: Search and analyze papers directly from arXiv and Semantic Scholar
- **Section Analysis**: Identifies and separates key paper sections (abstract, introduction, methods, etc.)
- **Key Concept Extraction**: Uses scientific NER to identify important concepts and entities
- **Research Gap Analysis**: Identifies potential research gaps and future work opportunities
- **Novelty Assessment**: Analyzes the paper's novel contributions and approaches
- **Interactive Q&A**: Allows natural language queries about the paper's content
- **Citation Network Visualization**: Creates interactive network graphs of paper citations
- **Batch Processing**: Analyze multiple papers simultaneously with parallel processing
- **Caching**: Optimizes performance and reduces API costs through intelligent caching
- **Figure & Table Extraction**: Automatically extracts figures and tables from PDFs for reference

## Project Structure

```
ScholarLens/
│── app.py                     # Streamlit UI implementation
│── document_processing.py     # PDF extraction & processing
│── rag_functions.py           # Vector store and LLM pipeline
│── academic_apis.py           # Academic database API integrations
│── reference_analyzer.py      # Reference extraction and analysis
│── config.py                  # Configuration settings
│── requirements.txt           # Project dependencies
│── improvements/              # Enhanced features
│   │── multi_model.py         # Support for multiple LLM providers
│   │── caching.py             # Redis and file-based caching
│   │── batch_processing.py    # Parallel paper processing
│   │── citation_network.py    # Network graphs of citations
│── .env.example               # Template for environment variables
│── .gitignore                 # Files to exclude from version control
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

5. Set up API keys:
   - Copy `.env.example` to `.env`
   - Add your API keys to the `.env` file (OpenAI, Anthropic, or Google)

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. If you haven't set the API key in the `.env` file, you'll be prompted to enter it in the application's Settings tab

4. You can:
   - Upload a single research paper PDF for detailed analysis
   - Search arXiv or Semantic Scholar for papers
   - Batch process multiple papers at once
   - Configure model and processing settings

5. Explore different analyses through the tabs:
   - Paper Structure: View different sections and a summary of the paper
   - Key Concepts: Explore important terms and concepts with visualizations
   - Research Gaps: Identify potential areas for future research
   - Novelty Analysis: Understand the paper's unique contributions
   - Top References: Analyze the most cited references in the paper
   - Citation Network: Visualize the paper's citation network
   - Media: View extracted figures and tables

6. Use the Q&A interface to ask specific questions about the paper

## Requirements

- Python 3.8+
- OpenAI, Anthropic, or Google API key
- 8GB+ RAM recommended
- Redis (optional, for enhanced caching)

## Technical Details

### Components

- **DocumentProcessor**: Enhanced PDF processing and text extraction
  - Uses PDFPlumber and PyMuPDF for comprehensive extraction
  - Extracts figures, tables, and text with improved accuracy
  - Implements semantic chunking for better context preservation

- **RAGPipeline**: Improved RAG implementation
  - Supports multiple model providers (OpenAI, Anthropic, Google)
  - Uses caching to reduce API costs and improve performance
  - Provides detailed analysis capabilities with specialized prompts

- **CitationNetwork**: Citation analysis and visualization
  - Creates interactive network graphs of paper citations
  - Identifies central papers and influential connections
  - Provides network statistics and centrality metrics

- **BatchProcessor**: Parallel processing of multiple papers
  - Processes papers in parallel for efficiency
  - Provides status tracking and exportable results
  - Supports different export formats (JSON, CSV, ZIP)

### Models Used

- **OpenAI**: GPT-4o, text-embedding-3-small/large
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Google**: Gemini Pro models

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- LangChain for the RAG implementation
- Streamlit for the UI framework
- OpenAI, Anthropic, and Google for language models
- SpaCy for NER capabilities
- NetworkX and Plotly for network visualization
- PyMuPDF for enhanced PDF extraction

## Roadmap

- [x] Integration with academic APIs (arXiv, Semantic Scholar)
- [x] Citation network visualization
- [x] Multi-model support (OpenAI, Anthropic, Google)
- [x] Batch processing capabilities
- [x] Caching system for performance optimization
- [x] Enhanced PDF extraction with figures and tables
- [ ] Translation support for non-English papers
- [ ] Domain-specific fine-tuned embeddings
- [ ] Collaborative features and shared workspaces