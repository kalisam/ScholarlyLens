"""
Environment setup utilities
"""
import os
from dotenv import load_dotenv
import sys

def load_environment():
    """Load environment variables from .env file and setup paths"""
    # Load environment variables from .env file (if it exists)
    load_dotenv()
    
    # Add the parent directory to sys.path to allow direct imports
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # Set up other environment configurations as needed
    # For example, configure logging
    import logging
    logging.basicConfig(level=logging.INFO)