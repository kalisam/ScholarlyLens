# ScholarLens Restructuring Guide

This guide explains how to migrate from the original ScholarLens structure to the new, more modular structure.

## New Structure Overview

The new structure organizes the code into logical modules:

```
scholarlens/
│── app.py                      # Main entry point (simplified)
│── utils/                      # Utility functions
│   │── __init__.py
│   │── environment.py          
│   └── streamlit_fix.py        
│
│── ui/                         # User interface components
│   │── __init__.py   
│   │── main_page.py            
│   │── upload_view.py          
│   │── search_view.py          
│   │── batch_view.py           
│   │── settings_view.py        
│   └── components/             
│       │── __init__.py
│       │── analysis_tabs.py    
│       │── citation_network.py 
│       │── reference_display.py
│       └── qa_interface.py     
│
│── core/                       # Business logic
│   │── __init__.py             
│   │── session.py              
│   │── pdf_processor.py        
│   │── search_service.py       
│   │── batch_service.py        
│   └── settings_service.py     
│
│── document_processing.py      # Original files (unchanged)
│── config.py                   
│── academic_apis.py            
│── reference_analyzer.py       
│── rag_functions.py            
└── improvements/               # Original directory (unchanged)
```

## Migration Steps

### 1. Set Up the New Structure

Run the provided setup script to create the necessary directories:

```bash
python setup_new_structure.py
```

### 2. Copy New Files to Their Locations

1. Copy each new Python file to its corresponding location in the new structure.
2. Make sure to maintain the original files that weren't refactored.

### 3. Fix the Streamlit Error

The original error was related to the `streamlit_fix.py` file trying to access non-existent attributes. The new implementation uses a different approach that doesn't rely on modifying Streamlit's internal API.

### 4. Update the Python Path

Make sure the `scholarlens` directory is in your Python path:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/scholarlens
```

Or add the parent directory to your Python path in the code.

### 5. Test the Application

Run the application with:

```bash
python -m scholarlens.app
```

## Key Fixes and Improvements

1. **Fixed Streamlit Error**: Replaced the direct patching of Streamlit's internal API with a more robust approach.

2. **Improved Structure**:
   - Separated UI components from business logic
   - Created clear service boundaries
   - Improved code organization and maintainability

3. **Reduced Circular Dependencies**:
   - Used proper imports to avoid circular dependencies
   - Added proper package initialization

4. **Enhanced Error Handling**:
   - Added more robust error handling throughout the application
   - Improved user feedback for errors

## Troubleshooting

If you encounter any issues after restructuring:

1. **Import Errors**: Check if the package structure is correct and imports are properly updated.

2. **Module Not Found**: Ensure the Python path includes the parent directory of the `scholarlens` package.

3. **Streamlit-related Errors**: If you encounter any Streamlit-related errors, check the `utils/streamlit_fix.py` implementation.

4. **API Key Issues**: Make sure your API keys are properly loaded from the environment variables or settings.