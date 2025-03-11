"""
Question-answering interface component for paper interaction
"""
import streamlit as st

def render_qa_interface():
    """Render the Q&A interface for asking questions about the paper"""
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