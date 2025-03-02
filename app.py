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
        
        # Display analysis tabs
        display_analysis_tabs(sections, full_text)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

def display_analysis_tabs(sections, full_text):
    """Display tabs for the different paper analyses."""
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Paper Structure", 
        "Key Concepts",
        "Research Gaps",
        "Novelty Analysis"
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
    tab_upload, tab_search = st.tabs(["Upload PDF", "Search Academic Databases"])
    
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
    
    # If a paper has been selected from search, process it
    if st.session_state.selected_paper:
        st.header(f"Analysis for: {st.session_state.selected_paper['title']}")
        process_uploaded_pdf(st.session_state.selected_paper['content'])
        # Clear the selected paper to avoid reprocessing
        st.session_state.selected_paper = None

if __name__ == "__main__":
    main()