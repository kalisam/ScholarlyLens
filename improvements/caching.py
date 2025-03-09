import redis
import hashlib
import json
import pickle
import os
import time
from typing import Any, Dict, List, Optional, Union, Callable
import logging
from functools import wraps

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Cache:
    """Abstract base class for caching implementations."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def set(self, key: str, value: Any, expiration: Optional[int] = None) -> None:
        """Set a value in the cache with optional expiration in seconds."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        raise NotImplementedError("Subclasses must implement this method")


class RedisCache(Cache):
    """Redis-based cache implementation."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, prefix: str = 'scholarlens:'):
        """
        Initialize the Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis DB number
            password: Redis password
            prefix: Key prefix to avoid collisions
        """
        try:
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False  # Keep as bytes for pickle compatibility
            )
            self.prefix = prefix
            # Test connection
            self.redis.ping()
            logger.info("Connected to Redis cache")
        except redis.ConnectionError as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.redis = None
    
    def _make_key(self, key: str) -> str:
        """Create a prefixed key to avoid collisions."""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if not self.redis:
            return None
        
        try:
            data = self.redis.get(self._make_key(key))
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting value from Redis: {e}")
            return None
    
    def set(self, key: str, value: Any, expiration: Optional[int] = None) -> None:
        """Set a value in the cache with optional expiration in seconds."""
        if not self.redis:
            return
        
        try:
            serialized = pickle.dumps(value)
            if expiration:
                self.redis.setex(self._make_key(key), expiration, serialized)
            else:
                self.redis.set(self._make_key(key), serialized)
        except Exception as e:
            logger.error(f"Error setting value in Redis: {e}")
    
    def delete(self, key: str) -> None:
        """Delete a value from the cache."""
        if not self.redis:
            return
        
        try:
            self.redis.delete(self._make_key(key))
        except Exception as e:
            logger.error(f"Error deleting value from Redis: {e}")
    
    def has(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        if not self.redis:
            return False
        
        try:
            return bool(self.redis.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Error checking key existence in Redis: {e}")
            return False


class FileCache(Cache):
    """File-based cache implementation for environments without Redis."""
    
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
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error getting value from file cache: {e}")
            return None
    
    def set(self, key: str, value: Any, expiration: Optional[int] = None) -> None:
        """Set a value in the cache with optional expiration in seconds."""
        path = self._get_cache_path(key)
        
        try:
            # Save the cache value
            with open(path, 'wb') as f:
                pickle.dump(value, f)
            
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


class CacheManager:
    """Manages caching functionality across the application."""
    
    def __init__(self, use_redis: bool = True, redis_host: str = 'localhost', 
                 redis_port: int = 6379, redis_password: Optional[str] = None,
                 file_cache_dir: str = '.cache/scholarlens'):
        """
        Initialize the cache manager.
        
        Args:
            use_redis: Whether to try using Redis
            redis_host: Redis host
            redis_port: Redis port
            redis_password: Redis password
            file_cache_dir: Directory for file cache
        """
        # Try to initialize Redis cache first if requested
        self.redis_cache = None
        if use_redis:
            try:
                self.redis_cache = RedisCache(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password
                )
                # Test connection
                if not self.redis_cache.redis:
                    logger.info("Redis not available, falling back to file cache")
                    self.redis_cache = None
            except Exception as e:
                logger.info(f"Redis not available, falling back to file cache: {e}")
                self.redis_cache = None
        
        # Initialize file cache as fallback
        self.file_cache = FileCache(cache_dir=file_cache_dir)
        
        # Set the active cache
        self.cache = self.redis_cache if self.redis_cache else self.file_cache
        logger.info(f"Using cache: {self.cache.__class__.__name__}")
    
    def cache_function(self, func=None, *, expiration: Optional[int] = None, 
                      key_prefix: str = '', include_args: bool = True):
        """
        Decorator to cache function results.
        
        Args:
            func: Function to decorate
            expiration: Cache expiration time in seconds (None for no expiration)
            key_prefix: Prefix for cache keys
            include_args: Whether to include function arguments in the cache key
            
        Returns:
            Decorated function
        """
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                # Create a cache key
                if include_args:
                    # Include arguments in the key
                    arg_key = hashlib.md5(pickle.dumps((args, kwargs))).hexdigest()
                    cache_key = f"{key_prefix}:{f.__name__}:{arg_key}"
                else:
                    # Just use the function name
                    cache_key = f"{key_prefix}:{f.__name__}"
                
                # Check if result is in cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_result
                
                # Execute the function
                result = f(*args, **kwargs)
                
                # Cache the result
                self.cache.set(cache_key, result, expiration)
                logger.debug(f"Cached result for {cache_key}")
                
                return result
            return wrapper
        
        if func:
            return decorator(func)
        return decorator
    
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
        """
        Clear all cache entries or those with a specific prefix.
        This is more reliable with Redis than file cache.
        """
        if isinstance(self.cache, RedisCache) and self.cache.redis:
            try:
                pattern = f"{self.cache.prefix}{key_prefix}*"
                cursor = 0
                while True:
                    cursor, keys = self.cache.redis.scan(cursor, pattern, 100)
                    if keys:
                        self.cache.redis.delete(*keys)
                    if cursor == 0:
                        break
                logger.info(f"Cleared all cache entries with prefix '{key_prefix}'")
            except Exception as e:
                logger.error(f"Error clearing Redis cache: {e}")
        elif isinstance(self.cache, FileCache):
            try:
                # This is a simplistic approach for file cache
                if key_prefix:
                    logger.warning("Clearing by prefix not fully supported with file cache")
                
                # Remove all files in cache directory
                for filename in os.listdir(self.cache.cache_dir):
                    os.remove(os.path.join(self.cache.cache_dir, filename))
                logger.info("Cleared all file cache entries")
            except Exception as e:
                logger.error(f"Error clearing file cache: {e}")


# Default cache manager instance with Redis disabled by default
cache_manager = CacheManager(use_redis=False)

def setup_redis_cache(host: str = 'localhost', port: int = 6379, 
                     password: Optional[str] = None) -> bool:
    """
    Setup the cache manager to use Redis.
    
    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        
    Returns:
        True if Redis was successfully set up, False otherwise
    """
    global cache_manager
    
    try:
        cache_manager = CacheManager(
            use_redis=True,
            redis_host=host,
            redis_port=port,
            redis_password=password
        )
        return isinstance(cache_manager.cache, RedisCache)
    except Exception as e:
        logger.error(f"Failed to setup Redis cache: {e}")
        return False