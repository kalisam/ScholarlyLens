"""
Batch processing view for ScholarLens
"""
import streamlit as st
import os
import base64
from core.batch_service import process_batch_files, get_batch_status, export_batch_results, clear_batch_results

def render_batch_view():
    """Render the batch processing interface"""
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
                process_batch_files(uploaded_files)
    
    with col2:
        # Display batch status
        st.subheader("Processing Status")
        
        status = get_batch_status()
        
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
                clear_batch_results()
                st.rerun()
    
    # Display results
    if hasattr(st.session_state, 'batch_manager') and st.session_state.batch_manager.batch_processor:
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
                output_path = export_batch_results(export_format)
                
                if output_path:
                    # Create download link
                    with open(output_path, 'rb' if export_format == 'zip' else 'r') as f:
                        file_data = f.read()
                    
                    b64_data = base64.b64encode(file_data if isinstance(file_data, bytes) else file_data.encode()).decode()
                    download_filename = os.path.basename(output_path)
                    mime_type = "application/zip" if export_format == "zip" else \
                            "text/csv" if export_format == "csv" else "application/json"
                    
                    href = f'<a href="data:{mime_type};base64,{b64_data}" download="{download_filename}">Download {export_format.upper()} File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            # Display individual results
            render_batch_results(results)

def render_batch_results(results):
    """Render batch processing results"""
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