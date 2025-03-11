"""
Streamlit compatibility fixes
This module applies patches to resolve compatibility issues between Streamlit and other libraries
"""
import sys
import types

# List of problematic modules known to cause issues with Streamlit
PROBLEMATIC_MODULES = [
    'torch._classes',
    'google._upb._message',
    'plotly.matplotlylib',
    'tensorflow',
]

def create_module_stub(name):
    """Create a stub module that won't cause issues"""
    module = types.ModuleType(name)
    module.__path__ = []  # Add an empty __path__ to simulate a package
    module.__file__ = '<stub>'
    return module

def patch_torch():
    """Fix the PyTorch issue with Streamlit"""
    try:
        import torch
        
        # Create a simple patch for torch._classes
        class ClassesFix:
            def __init__(self, original_module):
                self.original_module = original_module
            
            def __getattr__(self, attr):
                if attr == '__path__':
                    # Return a list to simulate a proper module path
                    from pathlib import Path
                    return [str(Path(torch.__file__).parent / '_classes')]
                return getattr(self.original_module, attr)
        
        # Apply the patch if torch is installed and loaded
        if hasattr(torch, '_classes'):
            torch._classes = ClassesFix(torch._classes)
            print("Applied PyTorch fix for Streamlit compatibility")
    except ImportError:
        # PyTorch not installed, no need to patch
        pass

def apply_fixes():
    """Apply all compatibility fixes"""
    # Apply PyTorch fix
    patch_torch()
    
    # Pre-create stubs for known problematic modules
    for module_name in PROBLEMATIC_MODULES:
        if module_name not in sys.modules:
            sys.modules[module_name] = create_module_stub(module_name)
    
    print("Applied Streamlit compatibility fixes")