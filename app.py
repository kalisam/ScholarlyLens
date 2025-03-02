import streamlit as st
import os
from document_processing import DocumentProcessor
from rag_functions import RAGPipeline
from config import ModelConfig

def init_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        st.session_state.config = ModelConfig()
        # Check for API key in environment variables
        if os.environ.get("OPENAI_API_KEY"):
            st.session_state.config.openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor(st.session_state.config)
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(st.session_state.config)

def main():
    st.title("ScholarLens: Tool for Paper Analysis")
    
    # Display API key input field if not set
    if not st.session_state.get('config', ModelConfig()).openai_api_key:
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        if api_key:
            if 'config' not in st.session_state:
                st.session_state.config = ModelConfig()
            st.session_state.config.openai_api_key = api_key
            # Reinitialize with the new API key
            st.session_state.processor = DocumentProcessor(st.session_state.config)
            st.session_state.rag_pipeline = RAGPipeline(st.session_state.config)
            st.success("API key set successfully!")
        else:
            st.warning("Please provide an OpenAI API key to continue")
            return
    
    # Initialize session state
    init_session_state()
    
    # Upload PDF
    upload_pdf_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

    if upload_pdf_file:
        try:
            # Process the PDF
            docs = st.session_state.processor.load_pdf(upload_pdf_file.getvalue())
            chunks = st.session_state.processor.create_chunks(docs, st.session_state.rag_pipeline.embeddings)
            
            # Create vector store
            st.session_state.rag_pipeline.create_vector_store(chunks)
            
            # Extract sections and full text
            sections = st.session_state.processor.extract_sections(docs)
            full_text = " ".join([doc.page_content for doc in docs])
            
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
                    gaps = st.session_state.rag_pipeline.identify_research_gaps(
                        sections['discussion'] + sections['conclusion']
                    )
                    st.write(gaps)
            
            with tab4:
                st.header("Novelty Assessment")
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

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
    else:
        st.info("Please upload a research paper (PDF) to begin analysis.")

if __name__ == "__main__":
    main()