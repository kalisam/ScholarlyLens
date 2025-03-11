"""
Citation network visualization component
"""
import streamlit as st

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