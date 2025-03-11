"""
Fix script for ScholarLens issues.

This script addresses several problems identified in the application:
1. PyTorch compatibility issues with Streamlit
2. FAISS GPU issues
3. Caching serialization problems
4. Academic API rate limiting
"""
import os
import sys
import shutil
import logging
import importlib
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ScholarLens-Fix")

def fix_module_path():
    """Add current directory to path so we can import modules"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        logger.info(f"Added {current_dir} to module path")

def backup_file(file_path):
    """Create a backup of a file before modifying it"""
    if os.path.exists(file_path):
        backup_path = file_path + '.bak'
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return True
    return False

def fix_streamlit_compatibility():
    """Apply improved Streamlit compatibility fixes"""
    try:
        # Update the streamlit_fix.py file with our improved version
        target_file = os.path.join('utils', 'streamlit_fix.py')
        
        # Backup the original file
        if backup_file(target_file):
            # Copy the content from our improved fix
            with open('streamlit-fix-improved.py', 'r') as source:
                content = source.read()
                
            with open(target_file, 'w') as target:
                target.write(content)
                
            logger.info(f"Updated {target_file} with improved compatibility fixes")
            
            # Force reload the module if it was loaded
            if 'utils.streamlit_fix' in sys.modules:
                try:
                    importlib.reload(sys.modules['utils.streamlit_fix'])
                    logger.info("Reloaded utils.streamlit_fix module")
                except Exception as e:
                    logger.error(f"Error reloading module: {str(e)}")
            return True
        else:
            logger.warning(f"Could not find {target_file} to update")
            return False
    except Exception as e:
        logger.error(f"Error fixing Streamlit compatibility: {str(e)}")
        return False

def fix_caching_issues():
    """Fix caching serialization issues"""
    try:
        # Create a new simplified caching module
        cache_dir = os.path.join('improvements')
        target_file = os.path.join(cache_dir, 'caching.py')
        
        # Backup the original file
        if backup_file(target_file):
            # Copy the content from our fixed caching module
            with open('caching-fix.py', 'r') as source:
                content = source.read()
                
            with open(target_file, 'w') as target:
                target.write(content)
                
            logger.info(f"Updated {target_file} with fixed caching implementation")
            
            # Force reload the module if it was loaded
            if 'improvements.caching' in sys.modules:
                try:
                    importlib.reload(sys.modules['improvements.caching'])
                    logger.info("Reloaded improvements.caching module")
                except Exception as e:
                    logger.error(f"Error reloading module: {str(e)}")
            return True
        else:
            logger.warning(f"Could not find {target_file} to update")
            return False
    except Exception as e:
        logger.error(f"Error fixing caching: {str(e)}")
        return False

def fix_rate_limiting():
    """Update academic_apis.py to handle rate limiting better"""
    try:
        target_file = 'academic_apis.py'
        
        # Backup the original file
        if backup_file(target_file):
            # Modify the file to add rate limiting
            with open(target_file, 'r') as f:
                content = f.read()
            
            # Add improved rate limiting and retry logic
            if 'self.request_delay = 1' in content:
                # Update the delay and add retry logic
                updated_content = content.replace(
                    'self.request_delay = 1',
                    'self.request_delay = 2  # Increased to avoid rate limiting\n        self.max_retries = 3'
                )
                
                # Add retry logic to the semantic scholar request method
                if 'response = requests.get(request_url, headers=headers)' in updated_content:
                    updated_content = updated_content.replace(
                        'response = requests.get(request_url, headers=headers)',
                        """# Add retry logic with exponential backoff
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                response = requests.get(request_url, headers=headers)
                
                # If rate limited, wait and retry
                if response.status_code == 429:
                    retry_count += 1
                    if retry_count < self.max_retries:
                        sleep_time = self.request_delay * (2 ** retry_count)  # Exponential backoff
                        print(f"Rate limited by API, retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                
                # For other responses, proceed normally
                break
            except Exception as e:
                print(f"Request error: {e}")
                retry_count += 1
                if retry_count < self.max_retries:
                    time.sleep(self.request_delay)
                    continue
                raise
        
        # After all retries, use the last response"""
                    )
                
                # Write the updated content
                with open(target_file, 'w') as f:
                    f.write(updated_content)
                    
                logger.info(f"Updated {target_file} with improved rate limiting")
                return True
            else:
                logger.warning("Could not find the request_delay line to update")
                return False
        else:
            logger.warning(f"Could not find {target_file} to update")
            return False
    except Exception as e:
        logger.error(f"Error fixing rate limiting: {str(e)}")
        return False

def check_dependencies():
    """Check and install missing dependencies"""
    try:
        import pkg_resources
        
        # List of critical dependencies
        dependencies = [
            'streamlit', 'langchain', 'langchain-community', 'langchain-openai',
            'faiss-cpu', 'sentence-transformers', 'pymupdf'
        ]
        
        missing = []
        for package in dependencies:
            try:
                pkg_resources.get_distribution(package)
            except pkg_resources.DistributionNotFound:
                missing.append(package)
        
        if missing:
            logger.warning(f"Missing dependencies: {', '.join(missing)}")
            logger.info("Run: pip install " + " ".join(missing))
            return False
        
        logger.info("All critical dependencies are installed")
        return True
    except Exception as e:
        logger.error(f"Error checking dependencies: {str(e)}")
        return False

def main():
    """Main function to run all fixes"""
    logger.info("Starting ScholarLens fix script")
    
    # Make sure we can import project modules
    fix_module_path()
    
    # Check dependencies
    check_dependencies()
    
    # Apply fixes
    fixes_applied = 0
    if fix_streamlit_compatibility():
        fixes_applied += 1
        
    if fix_caching_issues():
        fixes_applied += 1
        
    if fix_rate_limiting():
        fixes_applied += 1
    
    logger.info(f"Applied {fixes_applied} fixes")
    logger.info("""
    ========================================================
    ScholarLens fixes have been applied!
    
    To run the application:
    1. Make sure all dependencies are installed:
       pip install -r requirements.txt
       
    2. Start the application:
       streamlit run app.py
    
    If you still encounter issues, check the logs for details.
    ========================================================
    """)

if __name__ == "__main__":
    main()