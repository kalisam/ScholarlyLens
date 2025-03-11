"""
Academic search service for ScholarLens
"""
import streamlit as st

def search_papers(source, query):
    """
    Search for papers using the selected API.
    
    Args:
        source: API source to search ("arXiv" or "Semantic Scholar")
        query: The search query string
        
    Returns:
        List of paper dictionaries
    """
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
    """
    Display search results and allow selection.
    
    Args:
        papers: List of paper dictionaries
        source: Source of the papers ("arXiv" or "Semantic Scholar")
    """
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