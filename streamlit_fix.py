"""
This is a workaround for the Streamlit RuntimeError issue with PyTorch.
It modifies the Streamlit local_sources_watcher.py to skip problematic modules.
"""

import sys
import importlib
import types
import asyncio
import streamlit.watcher.local_sources_watcher as ls_watcher

# Original extract_paths function that causes issues
original_extract_paths = ls_watcher.extract_paths

# Define a safer extract_paths function
def safe_extract_paths(module):
    """Safely extract source file paths from a module, handling problematic modules."""
    if not isinstance(module, types.ModuleType):
        return []
    
    if not hasattr(module, "__file__") or not module.__file__:
        return []
    
    paths = [module.__file__]
    
    # Module has a __path__ attribute, so it's a package. Extract all paths from it.
    if hasattr(module, "__path__"):
        try:
            # Only use __path__ if it's a proper iterable, not a problematic object
            if isinstance(module.__path__, (list, tuple)) or hasattr(module.__path__, "__iter__"):
                for sub_path in module.__path__:
                    if isinstance(sub_path, str):
                        paths.append(sub_path)
                        
        except (RuntimeError, AttributeError, TypeError) as e:
            # Skip problematic modules like torch._classes
            print(f"Skipping problematic module path: {module.__name__}")
            return [module.__file__] if hasattr(module, "__file__") and module.__file__ else []
    
    return paths

# Patch Streamlit's extract_paths function
ls_watcher.extract_paths = safe_extract_paths

# Patch the get_module_paths function as well to handle RuntimeError in event loop
original_get_module_paths = ls_watcher.get_module_paths

def safe_get_module_paths():
    """Safely get all module paths, handling potential RuntimeError issues."""
    try:
        # First try the original method
        return original_get_module_paths()
    except RuntimeError as e:
        # If we get "no running event loop" error, create an event loop and run
        if "no running event loop" in str(e):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return original_get_module_paths()
            except Exception as inner_e:
                print(f"Error in event loop workaround: {inner_e}")
                # Fall back to a simple approach
                return []
        raise  # Re-raise if it's a different error

# Apply the patch
ls_watcher.get_module_paths = safe_get_module_paths

print("Applied Streamlit compatibility fixes")