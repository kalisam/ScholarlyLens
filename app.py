import streamlit as st
import os
import io
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

from document_processing import DocumentProcessor
from rag_functions import RAGPipeline
from config import ModelConfig
from academic_apis import AcademicAPIs
from reference_analyzer import ReferenceAnalyzer  # Import the new module

def init_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = ModelConfig()
        # Check for API key in environment variables
        if os.environ.get("OPENAI_API_KEY"):
            st.session_state.config.openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor(st.session_state.config)
    
    # Only initialize RAG pipeline if API key is available
    if 'rag_pipeline' not in st.session_state and st.session_state.config.openai_api_key:
        try:
            st.session_state.rag_pipeline = RAGPipeline(st.session_state.config)
        except Exception as e:
            st.error(f"Error initializing RAG pipeline: {e}")
            st.session_state.rag_pipeline = None
    
    if 'academic_apis' not in st.session_state:
        st.session_state.academic_apis = AcademicAPIs()
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'selected_paper' not in st.session_state:
        st.session_state.selected_paper = None
    if 'reference_analyzer' not in st.session_state:
        st.session_state.reference_analyzer = ReferenceAnalyzer(st.session_state.academic_apis)
    if 'top_references' not in st.session_state:
        st.session_state.top_references = []

def process_uploaded_pdf(pdf_content):
    """Process an uploaded or downloaded PDF file."""
    try:
        # Process the PDF
        docs = st.session_state.processor.load_pdf(pdf_content)
        chunks = st.session_state.processor.create_chunks(docs, st.session_state.rag_pipeline.embeddings)
        
        # Create vector store
        st.session_state.rag_pipeline.create_vector_store(chunks)
        
        # Extract sections and full text
        sections = st.session_state.processor.extract_sections(docs)
        full_text = " ".join([doc.page_content for doc in docs])
        
        # Extract top references
        top_references = st.session_state.reference_analyzer.get_most_cited_references(full_text)
        st.session_state.top_references = top_references
        
        # Display analysis tabs
        display_analysis_tabs(sections, full_text)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

def display_analysis_tabs(sections, full_text):
    """Display tabs for the different paper analyses."""
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Paper Structure", 
        "Key Concepts",
        "Research Gaps",
        "Novelty Analysis",
        "Top References"  # New tab
    ])
    
    with tab1:
        st.header("Paper Structure")
        for section, content in sections.items():
            if content.strip():
                with st.expander(f"{section.title()} Section"):
                    st.write(content)
    
    with tab2:
        st.header("Key Concepts")
        key_concepts = st.session_state.processor.identify_key_concepts(full_text)
        for concept_type, concepts in key_concepts.items():
            with st.expander(f"{concept_type}"):
                st.write(", ".join(concepts))
    
    with tab3:
        st.header("Research Gaps Analysis")
        if sections['discussion'] or sections['conclusion']:
            with st.spinner("Analyzing research gaps..."):
                gaps = st.session_state.rag_pipeline.identify_research_gaps(
                    sections['discussion'] + sections['conclusion']
                )
                st.write(gaps)
    
    with tab4:
        st.header("Novelty Assessment")
        with st.spinner("Assessing novelty..."):
            novelty = st.session_state.rag_pipeline.analyze_novelty(
                sections['abstract'] + sections['introduction'] + sections['conclusion']
            )
            st.write(novelty)
    
    with tab5:
        st.header("Most Cited References")
        display_top_references(st.session_state.top_references)

    # QA Interface
    st.header("Ask Questions")
    qa_chain = st.session_state.rag_pipeline.create_qa_chain()
    
    user_input = st.text_input("Ask a question about the paper:")
    if user_input:
        with st.spinner("Analyzing..."):
            try:
                response = qa_chain.run(user_input)
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def display_top_references(references):
    """Display the most cited references with detailed information."""
    if not references:
        st.info("No references found or extracted from the paper.")
        return
    
    st.write(f"Found {len(references)} most cited references in this paper:")
    
    for i, ref in enumerate(references):
        # Create a title that combines available information
        title_parts = []
        if 'title' in ref and ref['title']:
            title_parts.append(ref['title'])
        if 'year' in ref and ref['year']:
            title_parts.append(f"({ref['year']})")
        
        citation_count = ref.get('frequency', 0)
        title = " ".join(title_parts) if title_parts else f"Reference {i+1}"
        
        with st.expander(f"{i+1}. {title} - Cited {citation_count} times"):
            # Display author information
            if 'authors' in ref and ref['authors']:
                st.write(f"**Authors:** {', '.join(ref['authors'])}")
            
            # Display abstract if available
            if 'abstract' in ref and ref['abstract']:
                st.write(f"**Abstract:** {ref['abstract']}")
            
            # Display additional metadata
            metadata_items = []
            if 'year' in ref and ref['year']:
                metadata_items.append(f"**Year:** {ref['year']}")
            if 'ref_id' in ref and ref['ref_id']:
                metadata_items.append(f"**Reference ID:** {ref['ref_id']}")
            
            if metadata_items:
                st.write(" | ".join(metadata_items))
            
            # Add buttons for further actions
            col1, col2 = st.columns(2)
            
            with col1:
                if 'url' in ref and ref['url']:
                    st.markdown(f"[View on Semantic Scholar]({ref['url']})")
            
            with col2:
                if 'pdf_url' in ref and ref['pdf_url']:
                    if st.button(f"Analyze this reference", key=f"ref_{i}"):
                        with st.spinner("Downloading paper..."):
                            pdf_content = st.session_state.academic_apis.download_semantic_scholar_paper(ref['pdf_url'])
                            if pdf_content:
                                st.session_state.selected_paper = {
                                    'title': ref.get('title', f"Reference {i+1}"),
                                    'content': pdf_content
                                }
                                st.rerun()

def search_papers(source, query):
    """Search for papers using the selected API."""
    with st.spinner(f"Searching {source}..."):
        try:
            if source == "arXiv":
                return st.session_state.academic_apis.search_arxiv(query)
            elif source == "Semantic Scholar":
                return st.session_state.academic_apis.search_semantic_scholar(query)
            else:
                st.error("Invalid source selected")
                return []
        except Exception as e:
            st.error(f"Error searching {source}: {str(e)}")
            return []

def display_paper_results(papers, source):
    """Display search results and allow selection."""
    if not papers:
        st.info(f"No papers found on {source} for your query.")
        return
    
    st.write(f"Found {len(papers)} papers on {source}:")
    
    for i, paper in enumerate(papers):
        with st.expander(f"{i+1}. {paper['title']}"):
            st.write(f"**Authors:** {', '.join(paper['authors'])}")
            st.write(f"**Abstract:** {paper.get('abstract', 'No abstract available')}")
            
            if source == "arXiv":
                st.write(f"**Published:** {paper.get('published', 'N/A')}")
                st.write(f"**Categories:** {', '.join(paper.get('categories', []))}")
                if paper.get('pdf_url'):
                    if st.button(f"Analyze this paper", key=f"arxiv_{i}"):
                        with st.spinner("Downloading paper..."):
                            pdf_content = st.session_state.academic_apis.download_arxiv_paper(paper['arxiv_id'])
                            if pdf_content:
                                st.session_state.selected_paper = {
                                    'title': paper['title'],
                                    'content': pdf_content
                                }
                                st.rerun()
            
            elif source == "Semantic Scholar":
                st.write(f"**Year:** {paper.get('year', 'N/A')}")
                st.write(f"**Venue:** {paper.get('venue', 'N/A')}")
                st.write(f"**Citations:** {paper.get('citation_count', 'N/A')}")
                if paper.get('pdf_url'):
                    if st.button(f"Analyze this paper", key=f"semantic_{i}"):
                        with st.spinner("Downloading paper..."):
                            pdf_content = st.session_state.academic_apis.download_semantic_scholar_paper(paper['pdf_url'])
                            if pdf_content:
                                st.session_state.selected_paper = {
                                    'title': paper['title'],
                                    'content': pdf_content
                                }
                                st.rerun()

def main():
    st.title("ScholarLens: Tool for Paper Analysis")
    
    # Initialize session state
    init_session_state()
    
    # Display API key input field if not set
    if not st.session_state.get('config', ModelConfig()).openai_api_key:
        st.warning("OpenAI API key not found. Please enter it below to continue:")
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        if api_key:
            # Update the config
            if 'config' not in st.session_state:
                st.session_state.config = ModelConfig()
            st.session_state.config.openai_api_key = api_key
            # Initialize RAG pipeline with the new API key
            try:
                st.session_state.rag_pipeline = RAGPipeline(st.session_state.config)
                st.success("API key set successfully!")
                st.rerun()  # Rerun the app to initialize everything with the new API key
            except Exception as e:
                st.error(f"Error initializing with API key: {str(e)}")
        return  # Don't continue until API key is set
    
    # Create tabs for file upload and API search
    tab_upload, tab_search, tab_references = st.tabs([
        "Upload PDF", 
        "Search Academic Databases", 
        "Top References"  # New standalone tab
    ])
    
    with tab_upload:
        # Upload PDF
        upload_pdf_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

        if upload_pdf_file:
            process_uploaded_pdf(upload_pdf_file.getvalue())
    
    with tab_search:
        # Search for papers
        st.header("Search for Academic Papers")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("Enter search terms:")
        with col2:
            source = st.selectbox("Source:", ["arXiv", "Semantic Scholar"])
        
        if st.button("Search") and search_query:
            st.session_state.search_results = search_papers(source, search_query)
            display_paper_results(st.session_state.search_results, source)
    
    with tab_references:
        # Display top references from the analyzed paper
        st.header("Most Cited References")
        
        if st.session_state.top_references:
            display_top_references(st.session_state.top_references)
        else:
            st.info("Upload or select a paper first to see the most cited references.")
    
    # If a paper has been selected from search, process it
    if st.session_state.selected_paper:
        st.header(f"Analysis for: {st.session_state.selected_paper['title']}")
        process_uploaded_pdf(st.session_state.selected_paper['content'])
        # Clear the selected paper to avoid reprocessing
        st.session_state.selected_paper = None

if __name__ == "__main__":
    main()