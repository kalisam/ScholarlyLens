import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
import sys
import os

# Add the parent directory to sys.path to allow direct imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


class CitationNetwork:
    """Class for building and visualizing citation networks."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build_network_from_references(self, paper_title: str, references: List[Dict[str, Any]]) -> nx.DiGraph:
        """
        Build a citation network from a paper and its references.
        
        Args:
            paper_title: Title of the main paper
            references: List of reference dictionaries with 'title', 'authors', etc.
            
        Returns:
            NetworkX DiGraph representing the citation network
        """
        # Create a new graph
        self.graph = nx.DiGraph()
        
        # Add the main paper as the central node
        self.graph.add_node(paper_title, type='main', year=None, authors=None)
        
        # Add reference nodes and edges from the main paper to them
        for ref in references:
            ref_title = ref.get('title', 'Unknown reference')
            ref_year = ref.get('year')
            ref_authors = ref.get('authors', [])
            ref_citation_count = ref.get('frequency', 1)
            
            # Add the reference node
            self.graph.add_node(
                ref_title, 
                type='reference',
                year=ref_year,
                authors=ref_authors,
                citation_count=ref_citation_count
            )
            
            # Add edge from the main paper to the reference
            self.graph.add_edge(paper_title, ref_title, weight=ref_citation_count)
        
        return self.graph
    
    def build_extended_network(self, academic_apis, depth: int = 1) -> nx.DiGraph:
        """
        Extend the citation network by fetching citations of references.
        
        Args:
            academic_apis: AcademicAPIs instance to fetch citation data
            depth: How many levels deep to extend the network
            
        Returns:
            Extended NetworkX DiGraph
        """
        if depth <= 0 or not academic_apis:
            return self.graph
        
        # Only process reference nodes
        reference_nodes = [node for node, data in self.graph.nodes(data=True) 
                          if data.get('type') == 'reference']
        
        for ref_title in reference_nodes:
            # Skip if we've already processed this node
            if self.graph.out_degree(ref_title) > 0:
                continue
                
            # Search for this reference to get citations
            try:
                search_results = academic_apis.search_semantic_scholar(ref_title, limit=1)
                if not search_results:
                    continue
                    
                paper_id = search_results[0].get('semantic_scholar_id')
                if not paper_id:
                    continue
                
                # Get detailed information with citations
                paper_details = academic_apis.get_paper_details(paper_id)
                citations = paper_details.get('citations', [])
                
                # Limit to top N citations to avoid too large graphs
                for citation in citations[:5]:  # Top 5 citations
                    cited_title = citation.get('title', 'Unknown paper')
                    cited_authors = [author.get('name', '') for author in citation.get('authors', [])]
                    cited_year = citation.get('year')
                    
                    # Add the cited paper
                    self.graph.add_node(
                        cited_title,
                        type='citation',
                        year=cited_year,
                        authors=cited_authors
                    )
                    
                    # Add edge from reference to the cited paper
                    self.graph.add_edge(ref_title, cited_title, weight=1)
            except Exception as e:
                print(f"Error extending network for {ref_title}: {e}")
        
        # If depth > 1, recursively extend further (be careful with API rate limits)
        if depth > 1:
            # This is a simplified approach - in production, you'd want to be more careful
            # with API rate limits and large network sizes
            pass
            
        return self.graph
    
    def create_visualization(self, highlight_main: bool = True) -> Dict[str, Any]:
        """
        Create a Plotly visualization of the citation network.
        
        Args:
            highlight_main: Whether to highlight the main paper node
            
        Returns:
            Dictionary with Plotly figure and metadata
        """
        # Prepare node positions using a layout algorithm
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Prepare node attributes for visualization
        node_x = []
        node_y = []
        node_size = []
        node_color = []
        node_text = []
        
        for node, data in self.graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Set node size based on type or citations
            if data.get('type') == 'main':
                node_size.append(20)  # Main paper is larger
                node_color.append('red')
            elif data.get('type') == 'reference':
                # Size based on citation count in the paper
                citation_count = data.get('citation_count', 1)
                node_size.append(10 + citation_count * 3)
                node_color.append('blue')
            else:
                node_size.append(8)  # Other nodes (citations of references)
                node_color.append('green')
            
            # Prepare hover text
            authors_text = ', '.join(data.get('authors', [])[:3])
            if len(data.get('authors', [])) > 3:
                authors_text += '...'
                
            year_text = f", {data.get('year')}" if data.get('year') else ""
            hover_text = f"{node}{year_text}<br>{authors_text}"
            node_text.append(hover_text)
        
        # Create nodes trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=node_color,
                size=node_size,
                line=dict(width=1, color='#888')
            ),
            text=node_text
        )
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for u, v, data in self.graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            
            # Add the line coordinates
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge weight
            weight = data.get('weight', 1)
            edge_weights.append(weight)
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create the figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Citation Network',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        # Return the visualization data
        return {
            'figure': fig,
            'node_count': len(self.graph.nodes),
            'edge_count': len(self.graph.edges),
            'main_node': [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'main']
        }
    
    def identify_central_papers(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Identify the most central papers in the network.
        
        Args:
            top_n: Number of top papers to return
            
        Returns:
            List of dictionaries with paper information and centrality metrics
        """
        if len(self.graph.nodes) < 2:
            return []
            
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # Create combined ranking
        combined_centrality = {}
        for node in self.graph.nodes():
            combined_centrality[node] = degree_centrality.get(node, 0) + betweenness_centrality.get(node, 0)
        
        # Sort by combined centrality
        sorted_papers = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare results
        results = []
        for paper_title, centrality in sorted_papers[:top_n]:
            node_data = self.graph.nodes[paper_title]
            results.append({
                'title': paper_title,
                'type': node_data.get('type', 'unknown'),
                'year': node_data.get('year'),
                'authors': node_data.get('authors', []),
                'degree_centrality': degree_centrality.get(paper_title, 0),
                'betweenness_centrality': betweenness_centrality.get(paper_title, 0),
                'combined_centrality': centrality
            })
        
        return results
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        Calculate and return network statistics.
        
        Returns:
            Dictionary with network statistics
        """
        if len(self.graph.nodes) < 2:
            return {
                'node_count': len(self.graph.nodes),
                'edge_count': len(self.graph.edges),
                'density': 0,
                'avg_degree': 0
            }
            
        # Basic stats
        stats = {
            'node_count': len(self.graph.nodes),
            'edge_count': len(self.graph.edges),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.graph.nodes)
        }
        
        # Additional stats if the graph is large enough
        if len(self.graph.nodes) > 5:
            try:
                # These can fail on some graph structures
                stats['diameter'] = nx.diameter(self.graph.to_undirected())
                stats['avg_shortest_path'] = nx.average_shortest_path_length(self.graph.to_undirected())
            except:
                # Graph may be disconnected
                pass
        
        return stats