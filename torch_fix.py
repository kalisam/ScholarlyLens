"""
This is a fix for the PyTorch issue with Streamlit that causes:
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_
"""

import sys
import torch
from pathlib import Path

# Create a simple patch for torch._classes
class ClassesFix:
    def __init__(self, original_module):
        self.original_module = original_module
    
    def __getattr__(self, attr):
        if attr == '__path__':
            # Return a list to simulate a proper module path
            return [str(Path(torch.__file__).parent / '_classes')]
        return getattr(self.original_module, attr)

# Apply the patch
torch._classes = ClassesFix(torch._classes)

print("Applied PyTorch fix for Streamlit compatibility")