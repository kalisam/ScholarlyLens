"""
Reference display component for showing extracted references
"""
import streamlit as st

def display_top_references(references):
    """
    Display the most cited references with detailed information.
    
    Args:
        references: List of reference dictionaries
    """
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