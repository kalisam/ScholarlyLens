"""
Improved caching functionality for ScholarLens
"""
import os
import pickle
import logging
import time
import hashlib
import json
import io
import copy
import threading
from typing import Any, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersistentCacheHandler:
    """Handles safe serialization for caching complex objects"""
    
    @staticmethod
    def safe_copy(obj):
        """
        Creates a safe copy for serialization by removing unpicklable objects
        """
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        
        if isinstance(obj, (list, tuple)):
            return [PersistentCacheHandler.safe_copy(x) for x in obj]
        
        if isinstance(obj, dict):
            return {k: PersistentCacheHandler.safe_copy(v) for k, v in obj.items()}
        
        # Catch unpicklable objects like locks, threads, etc.
        try:
            # Test if object is picklable
            pickle.dumps(obj)
            return obj
        except (TypeError, pickle.PickleError):
            # Return a string representation if we can't pickle it
            return f"<Unpicklable:{obj.__class__.__name__}>"

    @staticmethod
    def serialize(obj):
        """Safely serialize an object for caching"""
        try:
            # Make a safe copy first
            safe_obj = PersistentCacheHandler.safe_copy(obj)
            return pickle.dumps(safe_obj)
        except Exception as e:
            logger.error(f"Serialization error: {str(e)}")
            # Return a minimal serialized object indicating the error
            return pickle.dumps({
                "_cache_error": True,
                "error_type": str(type(e).__name__),
                "error_msg": str(e)
            })
    
    @staticmethod
    def deserialize(data):
        """Deserialize cached data"""
        try:
            obj = pickle.loads(data)
            # Check if this is an error object
            if isinstance(obj, dict) and obj.get("_cache_error", False):
                logger.warning(f"Retrieved cache had error: {obj.get('error_msg')}")
                return None
            return obj
        except Exception as e:
            logger.error(f"Deserialization error: {str(e)}")
            return None

class FileCache:
    """Improved file-based cache with better error handling"""
    
    def __init__(self, cache_dir: str = '.cache/scholarlens'):
        """
        Initialize the file cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Using file cache at {cache_dir}")
    
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        # Use hash to avoid file system issues with special characters
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.pickle")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        path = self._get_cache_path(key)
        
        if not os.path.exists(path):
            return None
        
        try:
            # Check if the cache is expired (if metadata exists)
            metadata_path = path + '.meta'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if 'expiration' in metadata and metadata['expiration'] < time.time():
                        # Cache expired
                        self.delete(key)
                        return None
                        
            # Read the cached value
            with open(path, 'rb') as f:
                data = f.read()
                return PersistentCacheHandler.deserialize(data)
        except Exception as e:
            logger.error(f"Error getting value from file cache: {e}")
            return None
    
    def set(self, key: str, value: Any, expiration: Optional[int] = None) -> None:
        """Set a value in the cache with optional expiration in seconds."""
        path = self._get_cache_path(key)
        
        try:
            # Save the cache value with safe serialization
            with open(path, 'wb') as f:
                serialized = PersistentCacheHandler.serialize(value)
                f.write(serialized)
            
            # Save metadata if expiration is set
            if expiration:
                metadata_path = path + '.meta'
                with open(metadata_path, 'w') as f:
                    metadata = {
                        'key': key,
                        'created': time.time(),
                        'expiration': time.time() + expiration
                    }
                    json.dump(metadata, f)
        except Exception as e:
            logger.error(f"Error setting value in file cache: {e}")
    
    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        path = self._get_cache_path(key)
        metadata_path = path + '.meta'
        
        try:
            if os.path.exists(path):
                os.remove(path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
        except Exception as e:
            logger.error(f"Error deleting value from file cache: {e}")
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        path = self._get_cache_path(key)
        
        # Check if the cache file exists
        if not os.path.exists(path):
            return False
        
        # Check if the cache is expired (if metadata exists)
        metadata_path = path + '.meta'
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    if 'expiration' in metadata and metadata['expiration'] < time.time():
                        # Cache expired
                        return False
            except Exception:
                return False
                
        return True


# Create a simplified cache manager without Redis dependency for now
class SimpleCacheManager:
    """Simple cache manager using file cache only"""
    
    def __init__(self, cache_dir: str = '.cache/scholarlens'):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for file cache
        """
        self.cache = FileCache(cache_dir=cache_dir)
        logger.info(f"Using simplified cache manager with file cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, expiration: Optional[int] = None) -> None:
        """Set a value in the cache with optional expiration in seconds."""
        self.cache.set(key, value, expiration)
    
    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        self.cache.delete(key)
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return self.cache.has(key)
    
    def clear_all(self, key_prefix: str = '') -> None:
        """Clear all cache entries or those with a specific prefix."""
        try:
            count = 0
            for filename in os.listdir(self.cache.cache_dir):
                if filename.endswith('.pickle') and (not key_prefix or filename.startswith(key_prefix)):
                    os.remove(os.path.join(self.cache.cache_dir, filename))
                    # Also try to remove the metadata file
                    meta_path = os.path.join(self.cache.cache_dir, filename + '.meta')
                    if os.path.exists(meta_path):
                        os.remove(meta_path)
                    count += 1
            logger.info(f"Cleared {count} file cache entries")
        except Exception as e:
            logger.error(f"Error clearing file cache: {e}")


# Create a simple global cache manager instance
cache_manager = SimpleCacheManager()

# Expose this as the singleton for use in the rest of the application
def get_cache_manager():
    """Get the singleton cache manager instance"""
    return cache_manager

# Add a stub for Redis setup to maintain compatibility with existing code
def setup_redis_cache(host: str = 'localhost', port: int = 6379, 
                     password: Optional[str] = None) -> bool:
    """
    Stub function to maintain compatibility with code that expects Redis.
    Always returns False to indicate Redis is not available.
    
    Args:
        host: Redis host (ignored)
        port: Redis port (ignored)
        password: Redis password (ignored)
        
    Returns:
        False to indicate Redis is not being used
    """
    logger.warning("Redis support is disabled in this simplified version. Using file cache instead.")
    return False