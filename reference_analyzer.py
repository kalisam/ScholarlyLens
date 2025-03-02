import re
from collections import Counter
import datetime
from typing import List, Dict, Any, Optional

class ReferenceAnalyzer:
    """Class to extract and analyze references in academic papers."""
    
    def __init__(self, academic_apis=None):
        self.academic_apis = academic_apis
        
        # Regular expressions for different citation formats
        # Harvard style: (Author, Year)
        self.harvard_pattern = r'\(([A-Za-z\s-]+(?:et al\.)?),?\s+(\d{4}[a-z]?)\)'
        
        # IEEE style: [1], [2-5], etc.
        self.ieee_pattern = r'\[(\d+(?:-\d+)?(?:,\s*\d+(?:-\d+)?)*)\]'
        
        # APA style variations
        self.apa_pattern = r'([A-Za-z\s-]+(?:et al\.)?)\s+\((\d{4}[a-z]?)\)'
        
        # References section markers
        self.ref_section_markers = [
            r'references', r'bibliography', r'works cited', r'literature cited',
            r'cited literature', r'cited works'
        ]
    
    def extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract citations from the main text of a paper.
        
        Args:
            text: The full text of the paper
            
        Returns:
            List of citation dictionaries with format and reference ID
        """
        citations = []
        
        # Extract Harvard style citations
        harvard_matches = re.finditer(self.harvard_pattern, text)
        for match in harvard_matches:
            author = match.group(1).strip()
            year = match.group(2)
            citations.append({
                'format': 'harvard',
                'author': author,
                'year': year,
                'raw_citation': match.group(0),
                'frequency': 1  # Initialize count
            })
        
        # Extract IEEE style citations
        ieee_matches = re.finditer(self.ieee_pattern, text)
        for match in ieee_matches:
            ref_ids = match.group(1)
            # Handle ranges like [1-5]
            if '-' in ref_ids:
                start, end = map(int, ref_ids.split('-'))
                for ref_id in range(start, end + 1):
                    citations.append({
                        'format': 'ieee',
                        'ref_id': str(ref_id),
                        'raw_citation': f'[{ref_id}]',
                        'frequency': 1  # Initialize count
                    })
            else:
                # Handle comma-separated references like [1,2,3]
                for ref_id in re.findall(r'\d+', ref_ids):
                    citations.append({
                        'format': 'ieee',
                        'ref_id': ref_id,
                        'raw_citation': f'[{ref_id}]',
                        'frequency': 1  # Initialize count
                    })
        
        # Extract APA style citations
        apa_matches = re.finditer(self.apa_pattern, text)
        for match in apa_matches:
            author = match.group(1).strip()
            year = match.group(2)
            citations.append({
                'format': 'apa',
                'author': author,
                'year': year,
                'raw_citation': match.group(0),
                'frequency': 1  # Initialize count
            })
        
        return citations
    
    def extract_reference_section(self, text: str) -> str:
        """
        Extract the references section from the paper.
        
        Args:
            text: The full text of the paper
            
        Returns:
            The references section as a string
        """
        # Find the reference section using markers
        pattern = '|'.join(self.ref_section_markers)
        match = re.search(f'({pattern})', text, re.IGNORECASE)
        
        if match:
            start_pos = match.start()
            # References are typically at the end, so extract from the match to the end
            # or to the next major section if one exists
            section_headers = [
                r'\nappendix', r'\nacknowledgements', r'\nfunding', 
                r'\ndeclaration', r'\nauthor contributions'
            ]
            
            # Create pattern for end of references section
            end_pattern = '|'.join(section_headers)
            end_match = re.search(f'({end_pattern})', text[start_pos:], re.IGNORECASE)
            
            if end_match:
                end_pos = start_pos + end_match.start()
                return text[start_pos:end_pos]
            else:
                return text[start_pos:]
        
        return ""
    
    def parse_reference_list(self, ref_section: str) -> List[Dict[str, Any]]:
        """
        Parse the reference list into structured data.
        
        Args:
            ref_section: The references section text
            
        Returns:
            List of structured reference entries
        """
        references = []
        
        # Split into individual references
        # This is a simplified approach - may need refinement for different formats
        lines = ref_section.strip().split('\n')
        current_ref = ""
        ref_entries = []
        
        # Group lines into references
        for line in lines:
            # Skip the "References" header and empty lines
            if re.match(f"({'|'.join(self.ref_section_markers)})", line, re.IGNORECASE) or not line.strip():
                continue
                
            # Check if this is a new reference (starts with [number] or number.)
            if re.match(r'^\s*\[\d+\]|\s*\d+\.', line):
                if current_ref:  # Save the previous reference if it exists
                    ref_entries.append(current_ref)
                current_ref = line
            else:
                # Continue with the current reference
                current_ref += " " + line
        
        # Add the last reference
        if current_ref:
            ref_entries.append(current_ref)
        
        # Parse each reference entry
        for entry in ref_entries:
            ref = {}
            
            # Extract reference number for IEEE style
            num_match = re.match(r'^\s*\[(\d+)\]|\s*(\d+)\.', entry)
            if num_match:
                ref['ref_id'] = num_match.group(1) or num_match.group(2)
                
            # Try to extract title (usually in quotes or italics)
            title_match = re.search(r'"([^"]+)"|"([^"]+)"', entry)
            if title_match:
                ref['title'] = title_match.group(1) or title_match.group(2)
            else:
                # Fallback title extraction (may need improvement)
                ref['title'] = entry[:100] + "..." if len(entry) > 100 else entry
            
            # Extract year
            year_match = re.search(r'\((\d{4}[a-z]?)\)|\b(19|20)\d{2}[a-z]?\b', entry)
            if year_match:
                ref['year'] = year_match.group(1) or year_match.group(2)
            
            # Extract authors (simplified)
            # This is a challenging task and may require more sophistication
            if 'ref_id' in ref and ref['ref_id']:
                entry_without_id = re.sub(r'^\s*\[\d+\]|\s*\d+\.', '', entry).strip()
                # Assume authors come before the year or title
                potential_authors = entry_without_id.split(',')[0]
                ref['authors'] = [potential_authors]
            
            references.append(ref)
        
        return references
    
    def match_citations_to_references(self, citations: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match in-text citations to the reference list.
        
        Args:
            citations: List of extracted citations
            references: List of parsed references
            
        Returns:
            List of matched references with citation frequencies
        """
        # Count citation frequencies
        citation_counter = Counter()
        
        for citation in citations:
            if citation['format'] == 'ieee' and 'ref_id' in citation:
                citation_counter[citation['ref_id']] += 1
            elif 'author' in citation and 'year' in citation:
                # For Harvard and APA styles
                key = f"{citation['author']}_{citation['year']}"
                citation_counter[key] += 1
        
        # Match references to citation counts
        for ref in references:
            # For IEEE style
            if 'ref_id' in ref:
                ref['frequency'] = citation_counter.get(ref['ref_id'], 0)
            
            # For author-year styles (rough matching)
            elif 'authors' in ref and 'year' in ref:
                # Try to match with each author-year combination
                max_freq = 0
                for author in ref['authors']:
                    key = f"{author}_{ref['year']}"
                    freq = citation_counter.get(key, 0)
                    if freq > max_freq:
                        max_freq = freq
                ref['frequency'] = max_freq
            
            # Default if no match
            else:
                ref['frequency'] = 0
        
        return references
    
    def get_most_cited_references(self, text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most frequently cited references in a paper.
        
        Args:
            text: The full text of the paper
            limit: Maximum number of references to return
            
        Returns:
            List of the most cited references with additional metadata where available
        """
        # Extract citations and references
        citations = self.extract_citations_from_text(text)
        ref_section = self.extract_reference_section(text)
        references = self.parse_reference_list(ref_section)
        
        # Match citations to references and get frequencies
        matched_refs = self.match_citations_to_references(citations, references)
        
        # Sort by frequency (descending)
        sorted_refs = sorted(matched_refs, key=lambda x: x.get('frequency', 0), reverse=True)
        
        # Return the top references
        top_refs = sorted_refs[:limit]
        
        # Enhance reference metadata if academic_apis is available
        if self.academic_apis:
            enhanced_refs = []
            for ref in top_refs:
                if 'title' in ref:
                    # Try to get more metadata from APIs
                    try:
                        # Search for the paper by title
                        search_results = self.academic_apis.search_semantic_scholar(ref['title'], limit=1)
                        if search_results:
                            paper_data = search_results[0]
                            ref.update({
                                'abstract': paper_data.get('abstract', ''),
                                'url': paper_data.get('url', ''),
                                'pdf_url': paper_data.get('pdf_url', ''),
                                'semantic_scholar_id': paper_data.get('semantic_scholar_id', '')
                            })
                    except Exception as e:
                        print(f"Error enhancing reference metadata: {e}")
                
                enhanced_refs.append(ref)
            
            return enhanced_refs
        
        return top_refs

# Example usage
# analyzer = ReferenceAnalyzer()
# most_cited = analyzer.get_most_cited_references(paper_text)