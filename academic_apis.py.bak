import requests
import urllib.parse
import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import io

class AcademicAPIs:
    """Class to handle interactions with academic APIs like arXiv and Semantic Scholar."""
    
    def __init__(self):
        self.arxiv_base_url = "http://export.arxiv.org/api/query"
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"
        # Set a reasonable rate limit to avoid being blocked
        self.request_delay = 1  # seconds between requests

    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of papers with metadata
        """
        # URL encode the query
        encoded_query = urllib.parse.quote(query)
        
        # Construct the API request URL
        request_url = f"{self.arxiv_base_url}?search_query=all:{encoded_query}&start=0&max_results={max_results}"
        
        response = requests.get(request_url)
        
        if response.status_code != 200:
            raise Exception(f"arXiv API request failed with status code {response.status_code}")
        
        # Parse the XML response
        root = ET.fromstring(response.content)
        
        # Extract the papers from the response
        papers = []
        
        # Define the XML namespace
        namespace = {'arxiv': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('arxiv:entry', namespace):
            paper = {
                'title': entry.find('arxiv:title', namespace).text.strip(),
                'authors': [author.find('arxiv:name', namespace).text for author in entry.findall('arxiv:author', namespace)],
                'abstract': entry.find('arxiv:summary', namespace).text.strip(),
                'published': entry.find('arxiv:published', namespace).text,
                'updated': entry.find('arxiv:updated', namespace).text,
                'arxiv_url': next((link.get('href') for link in entry.findall('arxiv:link', namespace) 
                                  if link.get('type') == 'text/html'), None),
                'pdf_url': next((link.get('href') for link in entry.findall('arxiv:link', namespace) 
                                if link.get('title') == 'pdf'), None),
                'arxiv_id': entry.find('arxiv:id', namespace).text.split('/')[-1]
            }
            
            # Add categories/tags
            paper['categories'] = [category.get('term') for category in entry.findall('arxiv:category', namespace)]
            
            papers.append(paper)
            
        return papers
    
    def download_arxiv_paper(self, arxiv_id: str) -> Optional[bytes]:
        """
        Download a paper from arXiv by its ID.
        
        Args:
            arxiv_id: The arXiv ID of the paper
            
        Returns:
            PDF content as bytes or None if download fails
        """
        # Ensure we respect rate limits
        time.sleep(self.request_delay)
        
        # Construct the PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        try:
            response = requests.get(pdf_url)
            if response.status_code == 200:
                return response.content
            else:
                print(f"Failed to download PDF: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading PDF: {e}")
            return None
    
    def search_semantic_scholar(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar for papers matching the query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of papers with metadata
        """
        # URL encode the query
        encoded_query = urllib.parse.quote(query)
        
        # Construct the API request URL
        request_url = f"{self.semantic_scholar_base_url}/paper/search?query={encoded_query}&limit={limit}&fields=title,abstract,url,authors,year,venue,publicationDate,citationCount,openAccessPdf"
        
        # Include a reasonable user agent in the headers
        headers = {
            "User-Agent": "ScholarLens Research Tool/1.0",
        }
        
        response = requests.get(request_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Semantic Scholar API request failed with status code {response.status_code}")
        
        data = response.json()
        
        # Extract the papers from the response
        papers = []
        
        for item in data.get('data', []):
            if item is None:
                continue
                
            authors = []
            authors_list = item.get('authors', []) or []
            for author in authors_list:
                if author:
                    authors.append(author.get('name', ''))
            
            pdf_url = None
            open_access = item.get('openAccessPdf')
            if open_access and isinstance(open_access, dict):
                pdf_url = open_access.get('url')
            
            paper = {
                'title': item.get('title', ''),
                'authors': authors,
                'abstract': item.get('abstract', ''),
                'year': item.get('year'),
                'venue': item.get('venue', ''),
                'publication_date': item.get('publicationDate'),
                'citation_count': item.get('citationCount'),
                'url': item.get('url'),
                'pdf_url': pdf_url,
                'semantic_scholar_id': item.get('paperId')
            }
            
            papers.append(paper)
        
        return papers
    
    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a paper from Semantic Scholar.
        
        Args:
            paper_id: Semantic Scholar paper ID
            
        Returns:
            Detailed paper metadata
        """
        # Ensure we respect rate limits
        time.sleep(self.request_delay)
        
        # Construct the API request URL
        request_url = f"{self.semantic_scholar_base_url}/paper/{paper_id}?fields=title,abstract,authors,year,venue,publicationDate,citationCount,references,citations,openAccessPdf"
        
        # Include a reasonable user agent in the headers
        headers = {
            "User-Agent": "ScholarLens Research Tool/1.0",
        }
        
        response = requests.get(request_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Semantic Scholar API request failed with status code {response.status_code}")
        
        return response.json()
    
    def download_semantic_scholar_paper(self, url: str) -> Optional[bytes]:
        """
        Download a paper from a URL provided by Semantic Scholar.
        
        Args:
            url: URL to the PDF
            
        Returns:
            PDF content as bytes or None if download fails
        """
        # Ensure we respect rate limits
        time.sleep(self.request_delay)
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.content
            else:
                print(f"Failed to download PDF: Status code {response.status_code}")
                return None
        except Exception as e:
            print(f"Error downloading PDF: {e}")
            return None
            
    def get_top_cited_papers(self, field: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top cited papers from Semantic Scholar, optionally filtered by field.
        
        Args:
            field: Optional field/category to filter papers by
            limit: Maximum number of results to return
            
        Returns:
            List of top cited papers with metadata
        """
        # Construct the API request URL
        # Use the field as a query if provided, otherwise search for highly cited papers in general
        query = field if field else "machine learning artificial intelligence deep learning"
        encoded_query = urllib.parse.quote(query)
        
        # Request papers sorted by citation count in descending order
        request_url = f"{self.semantic_scholar_base_url}/paper/search?query={encoded_query}&limit={limit}&fields=title,abstract,url,authors,year,venue,publicationDate,citationCount,openAccessPdf&sort=citationCount:desc"
        
        # Include a reasonable user agent in the headers
        headers = {
            "User-Agent": "ScholarLens Research Tool/1.0",
        }
        
        response = requests.get(request_url, headers=headers)
        
        if response.status_code != 200:
            raise Exception(f"Semantic Scholar API request failed with status code {response.status_code}")
        
        data = response.json()
        
        # Extract the papers from the response
        papers = []
        
        for item in data.get('data', []):
            if item is None:
                continue
                
            authors = []
            authors_list = item.get('authors', []) or []
            for author in authors_list:
                if author:
                    authors.append(author.get('name', ''))
            
            pdf_url = None
            open_access = item.get('openAccessPdf')
            if open_access and isinstance(open_access, dict):
                pdf_url = open_access.get('url')
            
            paper = {
                'title': item.get('title', ''),
                'authors': authors,
                'abstract': item.get('abstract', ''),
                'year': item.get('year'),
                'venue': item.get('venue', ''),
                'publication_date': item.get('publicationDate'),
                'citation_count': item.get('citationCount'),
                'url': item.get('url'),
                'pdf_url': pdf_url,
                'semantic_scholar_id': item.get('paperId')
            }
            
            papers.append(paper)
        
        return papers