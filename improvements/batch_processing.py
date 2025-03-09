import os
import tempfile
import pandas as pd
import threading
import queue
import time
import datetime
import json
import zipfile
import io
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, TYPE_CHECKING, Type
import sys
import os

# Add the parent directory to sys.path to allow direct imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from document_processing import DocumentProcessor
from config import ModelConfig

# For type checking only
if TYPE_CHECKING:
    from rag_functions import RAGPipeline

# Break circular import - import RAGPipeline only when needed
def get_rag_pipeline():
    from rag_functions import RAGPipeline
    return RAGPipeline


class PaperAnalysisResult:
    """Class to store the analysis results for a single paper."""
    
    def __init__(self, paper_title: str, file_name: str):
        """
        Initialize a paper analysis result.
        
        Args:
            paper_title: Title of the paper
            file_name: Original file name
        """
        self.paper_title = paper_title
        self.file_name = file_name
        self.sections = {}
        self.key_concepts = {}
        self.research_gaps = ""
        self.novelty = ""
        self.top_references = []
        self.status = "pending"  # pending, processing, completed, failed
        self.error = None
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "paper_title": self.paper_title,
            "file_name": self.file_name,
            "sections": self.sections,
            "key_concepts": self.key_concepts,
            "research_gaps": self.research_gaps,
            "novelty": self.novelty,
            "top_references": self.top_references,
            "status": self.status,
            "error": str(self.error) if self.error else None,
            "timestamp": self.timestamp
        }
    
    def to_json(self) -> str:
        """Convert the result to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperAnalysisResult':
        """Create a result object from a dictionary."""
        result = cls(data["paper_title"], data["file_name"])
        result.sections = data.get("sections", {})
        result.key_concepts = data.get("key_concepts", {})
        result.research_gaps = data.get("research_gaps", "")
        result.novelty = data.get("novelty", "")
        result.top_references = data.get("top_references", [])
        result.status = data.get("status", "completed")
        result.error = data.get("error")
        result.timestamp = data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return result


class BatchProcessor:
    """Class to handle batch processing of multiple PDF papers."""
    
    def __init__(self, model_config: ModelConfig, num_workers: int = 2):
        """
        Initialize the batch processor.
        
        Args:
            model_config: Model configuration
            num_workers: Number of parallel workers
        """
        self.model_config = model_config
        self.num_workers = num_workers
        self.queue = queue.Queue()
        self.results = {}  # Map of file_name to PaperAnalysisResult
        self.workers = []
        self.running = False
        self.lock = threading.Lock()
        
    def start_workers(self):
        """Start worker threads."""
        if self.running:
            return
            
        self.running = True
        self.workers = []
        
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
    def stop_workers(self):
        """Stop worker threads."""
        self.running = False
        for _ in range(self.num_workers):
            self.queue.put(None)  # Signal workers to exit
            
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=1.0)
                
        self.workers = []
        
    def _worker_thread(self, worker_id: int):
        """
        Worker thread that processes papers from the queue.
        
        Args:
            worker_id: ID of the worker thread
        """
        # Initialize document processor and RAG pipeline
        processor = DocumentProcessor(self.model_config)
        RAGPipeline = get_rag_pipeline()
        rag_pipeline = RAGPipeline(self.model_config)
        
        while self.running:
            try:
                # Get a task from the queue (block with timeout to check running flag)
                task = self.queue.get(timeout=1.0)
                if task is None:
                    break
                    
                paper_content, file_name, paper_title, result_id = task
                
                # Update status to processing
                with self.lock:
                    self.results[result_id].status = "processing"
                
                # Process the paper
                self._process_paper(processor, rag_pipeline, paper_content, file_name, paper_title, result_id)
                
                # Mark the task as done
                self.queue.task_done()
                
            except queue.Empty:
                # No tasks available, continue
                continue
                
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                # If we had a task that failed, mark it as failed
                if 'result_id' in locals():
                    with self.lock:
                        if result_id in self.results:
                            self.results[result_id].status = "failed"
                            self.results[result_id].error = str(e)
                    
                    # Mark the task as done even if it failed
                    self.queue.task_done()
    
    def _process_paper(self, processor: DocumentProcessor, rag_pipeline: Any, 
                      paper_content: bytes, file_name: str, paper_title: str, result_id: str):
        """
        Process a single paper.
        
        Args:
            processor: Document processor
            rag_pipeline: RAG pipeline
            paper_content: PDF content
            file_name: Original file name
            paper_title: Paper title
            result_id: Result ID
        """
        try:
            # Process the PDF
            docs = processor.load_pdf(paper_content)
            chunks = processor.create_chunks(docs, rag_pipeline.embeddings)
            
            # Create vector store
            rag_pipeline.create_vector_store(chunks)
            
            # Extract sections and full text
            sections = processor.extract_sections(docs)
            full_text = " ".join([doc.page_content for doc in docs])
            
            # Extract key concepts
            key_concepts = processor.identify_key_concepts(full_text)
            
            # Analyze research gaps if the paper has discussion/conclusion sections
            research_gaps = ""
            if sections['discussion'] or sections['conclusion']:
                research_gaps = rag_pipeline.identify_research_gaps(
                    sections['discussion'] + sections['conclusion']
                )
            
            # Analyze novelty
            novelty = rag_pipeline.analyze_novelty(
                sections['abstract'] + sections['introduction'] + sections['conclusion']
            )
            
            # Update the result with extracted information
            with self.lock:
                result = self.results[result_id]
                result.sections = sections
                result.key_concepts = key_concepts
                result.research_gaps = research_gaps
                result.novelty = novelty
                result.status = "completed"
                
        except Exception as e:
            # Update the result with error information
            with self.lock:
                result = self.results[result_id]
                result.status = "failed"
                result.error = str(e)
                
            raise
    
    def add_paper(self, paper_content: bytes, file_name: str, paper_title: Optional[str] = None) -> str:
        """
        Add a paper to the processing queue.
        
        Args:
            paper_content: PDF content
            file_name: Original file name
            paper_title: Paper title (optional, will use file name if not provided)
            
        Returns:
            Result ID
        """
        # Generate a unique ID for this paper
        result_id = f"{file_name}_{int(time.time())}"
        
        # Use file name as title if not provided
        if not paper_title:
            paper_title = os.path.splitext(file_name)[0]
        
        # Create a result object
        result = PaperAnalysisResult(paper_title, file_name)
        
        # Store the result
        with self.lock:
            self.results[result_id] = result
        
        # Add to the processing queue
        self.queue.put((paper_content, file_name, paper_title, result_id))
        
        return result_id
    
    def get_result(self, result_id: str) -> Optional[PaperAnalysisResult]:
        """
        Get the result for a specific paper.
        
        Args:
            result_id: Result ID returned by add_paper
            
        Returns:
            PaperAnalysisResult or None if not found
        """
        with self.lock:
            return self.results.get(result_id)
    
    def get_all_results(self) -> Dict[str, PaperAnalysisResult]:
        """
        Get all results.
        
        Returns:
            Dictionary of result_id to PaperAnalysisResult
        """
        with self.lock:
            return self.results.copy()
    
    def export_results(self, format: str = "json") -> Union[str, bytes]:
        """
        Export all results in the specified format.
        
        Args:
            format: Export format ("json", "csv", "zip")
            
        Returns:
            String or bytes containing the exported data
        """
        with self.lock:
            results = self.results.copy()
        
        if format.lower() == "json":
            # Export as JSON
            data = {result_id: result.to_dict() for result_id, result in results.items()}
            return json.dumps(data, indent=2)
            
        elif format.lower() == "csv":
            # Export as CSV (flattened data)
            flattened_data = []
            for result_id, result in results.items():
                row = {
                    "result_id": result_id,
                    "paper_title": result.paper_title,
                    "file_name": result.file_name,
                    "status": result.status,
                    "timestamp": result.timestamp,
                    "error": str(result.error) if result.error else "",
                    "has_abstract": bool(result.sections.get("abstract", "")),
                    "has_introduction": bool(result.sections.get("introduction", "")),
                    "has_methods": bool(result.sections.get("methods", "")),
                    "has_results": bool(result.sections.get("results", "")),
                    "has_discussion": bool(result.sections.get("discussion", "")),
                    "has_conclusion": bool(result.sections.get("conclusion", "")),
                    "research_gaps": result.research_gaps,
                    "novelty": result.novelty,
                    "key_concepts_count": sum(len(concepts) for concepts in result.key_concepts.values())
                }
                flattened_data.append(row)
                
            # Convert to DataFrame and export as CSV
            df = pd.DataFrame(flattened_data)
            return df.to_csv(index=False)
            
        elif format.lower() == "zip":
            # Export as a ZIP file containing JSON files
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add a summary JSON
                summary = {
                    "total_papers": len(results),
                    "completed": sum(1 for r in results.values() if r.status == "completed"),
                    "failed": sum(1 for r in results.values() if r.status == "failed"),
                    "pending": sum(1 for r in results.values() if r.status == "pending"),
                    "processing": sum(1 for r in results.values() if r.status == "processing"),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                zip_file.writestr("summary.json", json.dumps(summary, indent=2))
                
                # Add individual result files
                for result_id, result in results.items():
                    # Sanitize file name to be safe for zip
                    safe_id = "".join(c if c.isalnum() or c in "._- " else "_" for c in result_id)
                    zip_file.writestr(f"results/{safe_id}.json", result.to_json())
            
            return zip_buffer.getvalue()
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_results(self):
        """Clear all results."""
        with self.lock:
            self.results.clear()


class BatchAnalysisManager:
    """High-level manager for batch analysis of papers."""
    
    def __init__(self, model_config: ModelConfig, result_dir: str = "batch_results"):
        """
        Initialize the batch analysis manager.
        
        Args:
            model_config: Model configuration
            result_dir: Directory to store results
        """
        self.model_config = model_config
        self.result_dir = result_dir
        self.batch_processor = None
        
        # Create result directory if it doesn't exist
        os.makedirs(result_dir, exist_ok=True)
    
    def start_batch_processor(self, num_workers: int = 2):
        """
        Start the batch processor.
        
        Args:
            num_workers: Number of parallel workers
        """
        if self.batch_processor:
            self.stop_batch_processor()
        
        self.batch_processor = BatchProcessor(self.model_config, num_workers)
        self.batch_processor.start_workers()
        
        return self.batch_processor
    
    def stop_batch_processor(self):
        """Stop the batch processor."""
        if self.batch_processor:
            self.batch_processor.stop_workers()
            self.batch_processor = None
    
    def process_directory(self, directory_path: str) -> List[str]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of result IDs
        """
        if not self.batch_processor:
            self.start_batch_processor()
        
        result_ids = []
        
        # Find all PDF files in the directory
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                
                # Read the PDF content
                with open(file_path, 'rb') as f:
                    pdf_content = f.read()
                
                # Add to batch processor
                result_id = self.batch_processor.add_paper(
                    pdf_content, 
                    filename,
                    os.path.splitext(filename)[0]  # Use filename without extension as title
                )
                
                result_ids.append(result_id)
        
        return result_ids
    
    def save_results(self, format: str = "json", filename: Optional[str] = None) -> str:
        """
        Save batch results to a file.
        
        Args:
            format: Export format ("json", "csv", "zip")
            filename: Optional filename (will generate one if not provided)
            
        Returns:
            Path to the saved file
        """
        if not self.batch_processor:
            raise ValueError("Batch processor not started")
        
        # Generate a filename if not provided
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_results_{timestamp}.{format}"
        
        # Full path to output file
        output_path = os.path.join(self.result_dir, filename)
        
        # Export and save results
        result_data = self.batch_processor.export_results(format)
        
        # Write to file (binary mode for ZIP, text mode for others)
        if format.lower() == "zip":
            with open(output_path, 'wb') as f:
                f.write(result_data)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result_data)
        
        return output_path
    
    def get_batch_status(self) -> Dict[str, Any]:
        """
        Get the current status of the batch process.
        
        Returns:
            Dictionary with batch status information
        """
        if not self.batch_processor:
            return {
                "status": "not_started",
                "queue_size": 0,
                "completed": 0,
                "failed": 0,
                "pending": 0,
                "processing": 0,
                "total": 0
            }
        
        results = self.batch_processor.get_all_results()
        
        return {
            "status": "running" if self.batch_processor.running else "stopped",
            "queue_size": self.batch_processor.queue.unfinished_tasks,
            "completed": sum(1 for r in results.values() if r.status == "completed"),
            "failed": sum(1 for r in results.values() if r.status == "failed"),
            "pending": sum(1 for r in results.values() if r.status == "pending"),
            "processing": sum(1 for r in results.values() if r.status == "processing"),
            "total": len(results)
        }