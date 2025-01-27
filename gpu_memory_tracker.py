import tensorflow as tf
import logging
import time
from contextlib import contextmanager
from typing import Optional

class GPUMemoryTracker:
    def __init__(self, log_file: Optional[str] = None):
        self.logger = self._setup_logger(log_file)
        
    def _setup_logger(self, log_file: Optional[str]) -> logging.Logger:
        logger = logging.getLogger('GPUMemoryTracker')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        return logger
    
    def _get_memory_info(self):
        try:
            return tf.config.experimental.get_memory_info('GPU:0')
        except:
            return {'peak': 0, 'current': 0}
    
    def log_memory(self, tag: str = ""):
        mem_info = self._get_memory_info()
        peak_mb = mem_info['peak'] / (1024 * 1024)
        current_mb = mem_info['current'] / (1024 * 1024)
        
        self.logger.info(
            f"[{tag}] Peak: {peak_mb:.2f}MB, Current: {current_mb:.2f}MB"
        )

    @contextmanager
    def track(self, operation_name: str):
        """Context manager for tracking memory during an operation"""
        self.log_memory(f"Before {operation_name}")
        try:
            yield
        finally:
            self.log_memory(f"After {operation_name}")

# Global instance for convenience
default_tracker = GPUMemoryTracker()

def track_memory(tag: str = ""):
    """Decorator for tracking memory usage in functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with default_tracker.track(tag or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator
