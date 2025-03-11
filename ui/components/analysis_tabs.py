"""
Analysis tabs component for displaying processed PDF analysis
"""
import streamlit as st
import pandas as pd
from ui.components.citation_network import display_citation_network
from ui.components.reference_display import display_top_references
from ui.components.qa_interface import render_qa_interface

def display_analysis_tabs(sections, full_text):
    """
    Display tabs for the different paper analyses.
    
    Args:
        sections: Dictionary of paper sections
        full_text: Complete text of the paper
    """
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
        display_paper_structure_tab(sections)
    
    with tab2:
        display_key_concepts_tab(full_text)
    
    with tab3:
        display_research_gaps_tab(sections)
    
    with tab4:
        display_novelty_tab(sections)
    
    with tab5:
        display_top_references(st.session_state.top_references)
    
    with tab6:
        display_citation_network()
        
    with tab7:
        display_media_tab()
    
    # Display Q&A Interface
    render_qa_interface()

def display_paper_structure_tab(sections):
    """Display the paper structure tab content"""
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

def display_key_concepts_tab(full_text):
    """Display the key concepts tab content"""
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

def display_research_gaps_tab(sections):
    """Display the research gaps tab content"""
    st.header("Research Gaps Analysis")
    if sections['discussion'] or sections['conclusion']:
        with st.spinner("Analyzing research gaps..."):
            gaps = st.session_state.rag_pipeline.identify_research_gaps(
                sections['discussion'] + sections['conclusion']
            )
            st.write(gaps)

def display_novelty_tab(sections):
    """Display the novelty assessment tab content"""
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

def display_media_tab():
    """Display the figures & tables tab content"""
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