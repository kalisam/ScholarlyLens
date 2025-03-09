# Fix Streamlit compatibility issues
try:
    from streamlit_fix import *
except ImportError as e:
    print(f"Error importing streamlit_fix: {e}")

import streamlit as st
import os
import io
import json
import tempfile
import time
import base64
from datetime import datetime
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
import sys

# Add the parent directory to sys.path to allow direct imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Load environment variables from .env file (if it exists)
load_dotenv()

# Import core modules
from document_processing import DocumentProcessor
from config import ModelConfig
from academic_apis import AcademicAPIs
from reference_analyzer import ReferenceAnalyzer

# Import RAGPipeline - after other imports to avoid circular references
from rag_functions import RAGPipeline

# Import improvements directly to avoid circular dependencies
import improvements.caching
import improvements.multi_model
import improvements.batch_processing
import improvements.citation_network

# Get specific classes/functions
ModelFactory = improvements.multi_model.ModelFactory
cache_manager = improvements.caching.cache_manager
setup_redis_cache = improvements.caching.setup_redis_cache
BatchAnalysisManager = improvements.batch_processing.BatchAnalysisManager
CitationNetwork = improvements.citation_network.CitationNetwork

def init_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = ModelConfig()
        # Check for API key in environment variables
        if os.environ.get("OPENAI_API_KEY"):
            st.session_state.config.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if os.environ.get("ANTHROPIC_API_KEY"):
            st.session_state.config.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if os.environ.get("GOOGLE_API_KEY"):
            st.session_state.config.google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor(st.session_state.config)
    
    # Only initialize RAG pipeline if API key is available for selected provider
    if 'rag_pipeline' not in st.session_state and has_valid_api_key(st.session_state.config):
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
    if 'citation_network' not in st.session_state:
        st.session_state.citation_network = CitationNetwork()
    if 'batch_manager' not in st.session_state:
        st.session_state.batch_manager = BatchAnalysisManager(st.session_state.config)
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Upload PDF"
    if 'figures' not in st.session_state:
        st.session_state.figures = []
    if 'tables' not in st.session_state:
        st.session_state.tables = []
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = ""

def has_valid_api_key(config: ModelConfig) -> bool:
    """Check if there's a valid API key for the selected provider."""
    if config.model_provider == "openai":
        return bool(config.openai_api_key)
    elif config.model_provider == "anthropic":
        return bool(config.anthropic_api_key)
    elif config.model_provider == "google":
        return bool(config.google_api_key)
    return False

def process_uploaded_pdf(pdf_content):
    """Process an uploaded or downloaded PDF file using improved pipeline."""
    try:
        # Show processing status
        status_placeholder = st.empty()
        status_placeholder.info("Processing PDF - Step 1/5: Extracting text and media...")
        
        # Process the PDF with improved extraction
        if st.session_state.config.extract_images or st.session_state.config.extract_tables:
            # Use PyMuPDF for enhanced extraction
            docs, figures, tables = st.session_state.processor.load_pdf_with_pymupdf(pdf_content)
            st.session_state.figures = figures
            st.session_state.tables = tables
        else:
            # Use standard extraction
            docs = st.session_state.processor.load_pdf(pdf_content)
            st.session_state.figures = []
            st.session_state.tables = []
        
        status_placeholder.info("Processing PDF - Step 2/5: Creating semantic chunks...")
        chunks = st.session_state.processor.create_chunks(docs, st.session_state.rag_pipeline.embeddings)
        
        status_placeholder.info("Processing PDF - Step 3/5: Building vector index...")
        # Create vector store
        st.session_state.rag_pipeline.create_vector_store(chunks)
        
        status_placeholder.info("Processing PDF - Step 4/5: Analyzing document structure...")
        # Extract sections and full text - use improved version if available
        if hasattr(st.session_state.processor, 'extract_sections_improved'):
            sections = st.session_state.processor.extract_sections_improved(docs)
        else:
            sections = st.session_state.processor.extract_sections(docs)
        
        full_text = " ".join([doc.page_content for doc in docs])
        
        status_placeholder.info("Processing PDF - Step 5/5: Identifying references...")
        # Extract top references
        top_references = st.session_state.reference_analyzer.get_most_cited_references(full_text)
        st.session_state.top_references = top_references
        
        # Generate citation network
        if top_references:
            paper_title = sections.get('title', 'Untitled Paper')
            if not paper_title or paper_title == 'Untitled Paper':
                # Try to extract title from first page
                first_page = docs[0].page_content if docs else ""
                lines = first_page.split('\n')
                if lines and lines[0].strip():
                    paper_title = lines[0].strip()
            
            # Build citation network
            st.session_state.citation_network.build_network_from_references(
                paper_title, top_references
            )
        
        # Generate paper summary if not cached
        summary = st.session_state.rag_pipeline.generate_paper_summary(
            sections['abstract'] + sections['introduction'] + sections['conclusion']
        )
        st.session_state.summary = summary
        
        # Generate paper recommendations
        recommendations = st.session_state.rag_pipeline.generate_paper_recommendations(
            sections['abstract'] + sections['introduction'] + sections['conclusion']
        )
        st.session_state.recommendations = recommendations
        
        # Clear status message
        status_placeholder.empty()
        
        # Display analysis tabs
        display_analysis_tabs(sections, full_text)
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

def display_analysis_tabs(sections, full_text):
    """Display tabs for the different paper analyses."""
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Paper Structure", 
        "Key Concepts",
        "Research Gaps",
        "Novelty Analysis",
        "Top References",
        "Citation Network",
        "Media"
    ])
    
    with tab1:
        st.header("Paper Structure")
        
        # Show paper summary
        if st.session_state.summary:
            st.subheader("Summary")
            st.write(st.session_state.summary)
            st.divider()
        
        # Show sections
        for section, content in sections.items():
            if content.strip():
                with st.expander(f"{section.title()} Section"):
                    st.write(content)
    
    with tab2:
        st.header("Key Concepts")
        
        # Use improved concept extraction if available
        if hasattr(st.session_state.processor, 'identify_key_concepts_improved'):
            improved_concepts = st.session_state.processor.identify_key_concepts_improved(full_text)
            for concept_type, concepts in improved_concepts.items():
                with st.expander(f"{concept_type} ({len(concepts)})"):
                    for concept in concepts:
                        st.markdown(f"**{concept['text']}** (mentioned {concept['frequency']} times)")
                        if 'contexts' in concept and concept['contexts']:
                            context_sample = concept['contexts'][0]
                            st.markdown(f"*Sample context:* {context_sample}")
                            st.divider()
        else:
            # Fall back to basic concept extraction
            key_concepts = st.session_state.processor.identify_key_concepts(full_text)
            for concept_type, concepts in key_concepts.items():
                with st.expander(f"{concept_type}"):
                    st.write(", ".join(concepts))
        
        # Show extracted keywords and phrases if available
        if hasattr(st.session_state.processor, 'extract_keywords_and_phrases'):
            st.subheader("Important Keywords and Phrases")
            keywords = st.session_state.processor.extract_keywords_and_phrases(full_text)
            
            # Create dataframe for visualization
            kw_data = []
            for kw in keywords:
                kw_data.append({
                    'text': kw['text'],
                    'type': kw['type'],
                    'count': kw['count'],
                    'score': kw['score']
                })
            
            if kw_data:
                df = pd.DataFrame(kw_data)
                # Create a bar chart of top keywords by count
                fig = px.bar(
                    df.head(15), 
                    x='text', 
                    y='score',
                    color='type',
                    title='Top Keywords and Phrases by Relevance',
                    labels={'text': 'Term', 'score': 'Relevance Score', 'type': 'Type'}
                )
                st.plotly_chart(fig)
    
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
            
        # Show recommendations if available
        if st.session_state.recommendations:
            st.subheader("Related Paper Recommendations")
            st.write(st.session_state.recommendations)
    
    with tab5:
        st.header("Most Cited References")
        display_top_references(st.session_state.top_references)
    
    with tab6:
        st.header("Citation Network")
        display_citation_network()
        
    with tab7:
        st.header("Figures & Tables")
        
        # Display figures
        if st.session_state.figures:
            st.subheader(f"Extracted Figures ({len(st.session_state.figures)})")
            for i, figure in enumerate(st.session_state.figures):
                with st.expander(f"Figure {i+1}: {figure.caption}"):
                    st.image(figure.image_data, caption=figure.caption)
                    st.text(f"Page: {figure.page_num + 1}")
        else:
            st.info("No figures extracted from this document")
            
        # Display tables
        if st.session_state.tables:
            st.subheader(f"Extracted Tables ({len(st.session_state.tables)})")
            for i, table in enumerate(st.session_state.tables):
                with st.expander(f"Table {i+1}: {table.caption}"):
                    # Convert to dataframe for better display
                    if table.data and len(table.data) > 0:
                        # Use first row as header if it seems like a header row
                        if len(table.data) > 1:
                            df = pd.DataFrame(table.data[1:], columns=table.data[0])
                        else:
                            df = pd.DataFrame(table.data)
                        st.dataframe(df)
                    else:
                        st.write("Empty table")
                    st.text(f"Page: {table.page_num + 1}")
        else:
            st.info("No tables extracted from this document")

    # QA Interface
    st.header("Ask Questions")
    qa_chain = st.session_state.rag_pipeline.create_qa_chain()
    
    user_input = st.text_input("Ask a question about the paper:")
    if user_input:
        with st.spinner("Analyzing..."):
            try:
                # Use cached query response if available
                response = st.session_state.rag_pipeline.run_query(user_input)
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

def display_citation_network():
    """Display the citation network visualization."""
    if not hasattr(st.session_state, 'citation_network') or not st.session_state.citation_network.graph:
        st.info("No citation network available. Analyze a paper with references first.")
        return
    
    # Get network visualization 
    network_viz = st.session_state.citation_network.create_visualization()
    
    # Display network statistics
    stats = st.session_state.citation_network.get_network_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Papers", stats['node_count'])
    with col2:
        st.metric("Citations", stats['edge_count'])
    with col3:
        st.metric("Network Density", f"{stats['density']:.4f}")
    
    # Display the network visualization
    st.plotly_chart(network_viz['figure'], use_container_width=True)
    
    # Get and display central papers
    central_papers = st.session_state.citation_network.identify_central_papers()
    if central_papers:
        st.subheader("Most Central Papers in the Network")
        for i, paper in enumerate(central_papers):
            with st.expander(f"{i+1}. {paper['title']}"):
                st.write(f"**Type:** {paper['type']}")
                if paper['year']:
                    st.write(f"**Year:** {paper['year']}")
                if paper['authors']:
                    st.write(f"**Authors:** {', '.join(paper['authors'])}")
                st.write(f"**Centrality Score:** {paper['combined_centrality']:.4f}")

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

def display_batch_processing_ui():
    """Display batch processing interface."""
    st.header("Batch Process Multiple Papers")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader for multiple PDFs
        uploaded_files = st.file_uploader(
            "Upload multiple research papers (PDF)", 
            type="pdf",
            accept_multiple_files=True
        )
        
        # Process files button
        if uploaded_files:
            if st.button("Process All Files"):
                # Start batch processor if not already running
                if not st.session_state.batch_manager.batch_processor or \
                   not st.session_state.batch_manager.batch_processor.running:
                    st.session_state.batch_manager.start_batch_processor(
                        num_workers=st.session_state.config.batch_workers
                    )
                
                # Process each file
                result_ids = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Adding file {i+1}/{len(uploaded_files)}: {file.name}")
                    progress_bar.progress((i + 0.5) / len(uploaded_files))
                    
                    # Get file content
                    pdf_content = file.getvalue()
                    
                    # Add to batch processor
                    result_id = st.session_state.batch_manager.batch_processor.add_paper(
                        pdf_content, file.name
                    )
                    result_ids.append(result_id)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text(f"Added {len(uploaded_files)} files to processing queue")
                progress_bar.empty()
                
                st.success(f"Started processing {len(uploaded_files)} files in the background")
    
    with col2:
        # Display batch status
        st.subheader("Processing Status")
        
        status = st.session_state.batch_manager.get_batch_status()
        
        st.write(f"Status: {status['status']}")
        st.write(f"Queue size: {status['queue_size']}")
        st.write(f"Completed: {status['completed']}/{status['total']}")
        st.write(f"Failed: {status['failed']}")
        st.write(f"Processing: {status['processing']}")
        
        # Progress bar
        if status['total'] > 0:
            progress = status['completed'] / status['total']
            st.progress(progress)
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if status['status'] == 'running':
                if st.button("Stop Processing"):
                    st.session_state.batch_manager.stop_batch_processor()
                    st.rerun()
            else:
                if st.button("Start Processing"):
                    st.session_state.batch_manager.start_batch_processor(
                        num_workers=st.session_state.config.batch_workers
                    )
                    st.rerun()
        
        with col2:
            if st.button("Clear Results"):
                if st.session_state.batch_manager.batch_processor:
                    st.session_state.batch_manager.batch_processor.clear_results()
                    st.rerun()
    
    # Display results
    if st.session_state.batch_manager.batch_processor:
        results = st.session_state.batch_manager.batch_processor.get_all_results()
        
        if results:
            st.subheader("Batch Results")
            
            # Export options
            export_format = st.selectbox(
                "Export format", 
                ["json", "csv", "zip"],
                index=0
            )
            
            if st.button("Export Results"):
                try:
                    output_path = st.session_state.batch_manager.save_results(
                        format=export_format
                    )
                    
                    # Create download link
                    with open(output_path, 'rb' if export_format == 'zip' else 'r') as f:
                        file_data = f.read()
                    
                    b64_data = base64.b64encode(file_data if isinstance(file_data, bytes) else file_data.encode()).decode()
                    download_filename = os.path.basename(output_path)
                    mime_type = "application/zip" if export_format == "zip" else \
                               "text/csv" if export_format == "csv" else "application/json"
                    
                    href = f'<a href="data:{mime_type};base64,{b64_data}" download="{download_filename}">Download {export_format.upper()} File</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error exporting results: {str(e)}")
            
            # Display individual results
            for result_id, result in results.items():
                status_color = {
                    'completed': '🟢',
                    'processing': '🟠',
                    'pending': '⚪',
                    'failed': '🔴'
                }.get(result.status, '⚪')
                
                with st.expander(f"{status_color} {result.paper_title} ({result.file_name})"):
                    st.write(f"**Status:** {result.status}")
                    st.write(f"**Timestamp:** {result.timestamp}")
                    
                    if result.status == 'completed':
                        # Display tabs for details
                        tab1, tab2, tab3 = st.tabs(["Summary", "Key Concepts", "Research Gaps"])
                        
                        with tab1:
                            # Show sections
                            for section, content in result.sections.items():
                                if content.strip():
                                    with st.expander(f"{section.title()} Section"):
                                        st.write(content)
                        
                        with tab2:
                            # Show key concepts
                            for concept_type, concepts in result.key_concepts.items():
                                st.write(f"**{concept_type}:** {', '.join(concepts)}")
                        
                        with tab3:
                            # Show research gaps and novelty
                            st.write("**Research Gaps:**")
                            st.write(result.research_gaps)
                            st.write("**Novelty Analysis:**")
                            st.write(result.novelty)
                    
                    elif result.status == 'failed':
                        st.error(f"Error: {result.error}")

def display_settings_ui():
    """Display settings interface."""
    st.header("Settings")
    
    # Create tabs for different settings
    tab1, tab2, tab3, tab4 = st.tabs([
        "API Keys", 
        "Model Settings",
        "Processing Settings",
        "Cache Settings"
    ])
    
    with tab1:
        st.subheader("API Keys")
        
        # OpenAI API key
        openai_key = st.text_input(
            "OpenAI API Key", 
            value=st.session_state.config.openai_api_key,
            type="password"
        )
        
        # Anthropic API key
        anthropic_key = st.text_input(
            "Anthropic API Key", 
            value=st.session_state.config.anthropic_api_key,
            type="password"
        )
        
        # Google API key
        google_key = st.text_input(
            "Google API Key", 
            value=st.session_state.config.google_api_key,
            type="password"
        )
        
        # Update API keys
        if st.button("Save API Keys"):
            st.session_state.config.openai_api_key = openai_key
            st.session_state.config.anthropic_api_key = anthropic_key
            st.session_state.config.google_api_key = google_key
            
            # Reinitialize RAG pipeline with new keys
            if has_valid_api_key(st.session_state.config):
                try:
                    st.session_state.rag_pipeline = RAGPipeline(st.session_state.config)
                    st.success("API keys updated and RAG pipeline reinitialized!")
                except Exception as e:
                    st.error(f"Error initializing RAG pipeline: {e}")
            else:
                st.warning("No valid API key provided for the selected model provider.")
    
    with tab2:
        st.subheader("Model Settings")
        
        # Model provider
        model_provider = st.selectbox(
            "Model Provider",
            ["openai", "anthropic", "google"],
            index=["openai", "anthropic", "google"].index(st.session_state.config.model_provider)
        )
        
        # Available models based on provider
        available_models = {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "google": ["gemini-pro", "gemini-1.5-pro"]
        }.get(model_provider, ["gpt-4o"])
        
        # Default model
        default_model = {
            "openai": "gpt-4o",
            "anthropic": "claude-3-haiku-20240307",
            "google": "gemini-pro"
        }.get(model_provider, "gpt-4o")
        
        # Current model or default
        current_model = st.session_state.config.llm_model
        if current_model not in available_models:
            current_model = default_model
        
        # LLM model
        llm_model = st.selectbox(
            "Language Model",
            available_models,
            index=available_models.index(current_model) if current_model in available_models else 0
        )
        
        # Embedding model (only for OpenAI)
        embedding_model = st.selectbox(
            "Embedding Model",
            ["text-embedding-3-small", "text-embedding-3-large"],
            index=["text-embedding-3-small", "text-embedding-3-large"].index(
                st.session_state.config.embedding_model
            ) if st.session_state.config.embedding_model in ["text-embedding-3-small", "text-embedding-3-large"] else 0
        )
        
        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.config.temperature,
            step=0.1
        )
        
        # Max tokens
        max_tokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=st.session_state.config.max_tokens,
            step=256
        )
        
        # Update model settings
        if st.button("Save Model Settings"):
            needs_reinit = (
                model_provider != st.session_state.config.model_provider or
                llm_model != st.session_state.config.llm_model or
                embedding_model != st.session_state.config.embedding_model
            )
            
            # Update config
            st.session_state.config.model_provider = model_provider
            st.session_state.config.llm_model = llm_model
            st.session_state.config.embedding_model = embedding_model
            st.session_state.config.temperature = temperature
            st.session_state.config.max_tokens = max_tokens
            
            # Reinitialize RAG pipeline if needed
            if needs_reinit and has_valid_api_key(st.session_state.config):
                try:
                    st.session_state.rag_pipeline = RAGPipeline(st.session_state.config)
                    st.success("Model settings updated and RAG pipeline reinitialized!")
                except Exception as e:
                    st.error(f"Error initializing RAG pipeline: {e}")
            else:
                st.success("Model settings updated!")
    
    with tab3:
        st.subheader("Processing Settings")
        
        # Chunk size
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=st.session_state.config.chunk_size,
            step=100
        )
        
        # Chunk overlap
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=st.session_state.config.chunk_overlap,
            step=50
        )
        
        # Search k
        search_k = st.slider(
            "Number of search results (k)",
            min_value=1,
            max_value=10,
            value=st.session_state.config.search_k,
            step=1
        )
        
        # Extract images and tables
        extract_images = st.checkbox(
            "Extract images from PDFs",
            value=st.session_state.config.extract_images
        )
        
        extract_tables = st.checkbox(
            "Extract tables from PDFs",
            value=st.session_state.config.extract_tables
        )
        
        # Batch workers
        batch_workers = st.slider(
            "Batch processing workers",
            min_value=1,
            max_value=8,
            value=st.session_state.config.batch_workers,
            step=1
        )
        
        # Update processing settings
        if st.button("Save Processing Settings"):
            st.session_state.config.chunk_size = chunk_size
            st.session_state.config.chunk_overlap = chunk_overlap
            st.session_state.config.search_k = search_k
            st.session_state.config.extract_images = extract_images
            st.session_state.config.extract_tables = extract_tables
            st.session_state.config.batch_workers = batch_workers
            
            st.success("Processing settings updated!")
    
    with tab4:
        st.subheader("Cache Settings")
        
        # Use cache
        use_cache = st.checkbox(
            "Enable caching",
            value=st.session_state.config.use_cache
        )
        
        # Cache TTL
        cache_ttl = st.slider(
            "Cache TTL (seconds)",
            min_value=3600,
            max_value=604800,  # 1 week
            value=st.session_state.config.cache_ttl,
            step=3600
        )
        
        # Use Redis
        use_redis = st.checkbox(
            "Use Redis cache (instead of file cache)",
            value=st.session_state.config.use_redis
        )
        
        # Redis settings
        redis_host = st.text_input(
            "Redis Host",
            value=st.session_state.config.redis_host
        )
        
        redis_port = st.number_input(
            "Redis Port",
            value=st.session_state.config.redis_port,
            min_value=1,
            max_value=65535
        )
        
        redis_password = st.text_input(
            "Redis Password",
            value=st.session_state.config.redis_password,
            type="password"
        )
        
        # Update cache settings
        if st.button("Save Cache Settings"):
            st.session_state.config.use_cache = use_cache
            st.session_state.config.cache_ttl = cache_ttl
            st.session_state.config.use_redis = use_redis
            st.session_state.config.redis_host = redis_host
            st.session_state.config.redis_port = redis_port
            st.session_state.config.redis_password = redis_password
            
            # Setup Redis cache if enabled
            if use_cache and use_redis:
                from improvements.caching import setup_redis_cache
                redis_success = setup_redis_cache(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password
                )
                if redis_success:
                    st.success("Cache settings updated and Redis cache connected!")
                else:
                    st.warning("Cache settings updated but Redis connection failed. Using file cache.")
            else:
                st.success("Cache settings updated!")
        
        # Clear cache button
        if st.button("Clear Cache"):
            if hasattr(st.session_state, 'rag_pipeline'):
                cleared = st.session_state.rag_pipeline.clear_cache()
                if cleared:
                    st.success("Cache cleared successfully!")
                else:
                    st.warning("Cache is not enabled or could not be cleared.")

def main():
    st.title("ScholarLens: Advanced Research Paper Analysis")
    
    # Initialize session state
    init_session_state()
    
    # Create main navigation
    with st.sidebar:
        selected_tab = st.sidebar.radio(
            "Navigation",
            ["Upload PDF", "Search Academic DB", "Batch Processing", "Settings"],
            index=0 if 'current_tab' not in st.session_state else 
                ["Upload PDF", "Search Academic DB", "Batch Processing", "Settings"].index(st.session_state.current_tab)
        )
        
        st.session_state.current_tab = selected_tab
        
        # Display API status
        st.subheader("API Status")
        provider = st.session_state.config.model_provider
        has_api = has_valid_api_key(st.session_state.config)
        
        if has_api:
            st.success(f"✅ {provider.capitalize()} API key set")
        else:
            st.error(f"❌ No {provider.capitalize()} API key")
            st.warning("Go to Settings to add your API key")
    
    # Display API key input field if not set
    if not has_valid_api_key(st.session_state.config):
        st.warning(f"No API key found for {st.session_state.config.model_provider.capitalize()} Provider. Please go to Settings and add your API key to continue.")
        
        # Display settings if on main page
        if selected_tab not in ["Search Academic DB", "Settings"]:
            display_settings_ui()
            return
    
    # Display content based on selected tab
    if selected_tab == "Upload PDF":
        # Upload PDF
        upload_pdf_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

        if upload_pdf_file:
            process_uploaded_pdf(upload_pdf_file.getvalue())
    
    elif selected_tab == "Search Academic DB":
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
    
    elif selected_tab == "Batch Processing":
        display_batch_processing_ui()
    
    elif selected_tab == "Settings":
        display_settings_ui()
    
    # If a paper has been selected from search, process it
    if st.session_state.selected_paper:
        st.header(f"Analysis for: {st.session_state.selected_paper['title']}")
        process_uploaded_pdf(st.session_state.selected_paper['content'])
        # Clear the selected paper to avoid reprocessing
        st.session_state.selected_paper = None

if __name__ == "__main__":
    main()