import time
from functools import wraps
from typing import Callable, Any
import numpy as np
import pandas as pd

def time_imputation(name: str) -> Callable:
    """Decorator to time imputation methods."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Store timing in the instance
            self = args[0]  # The first argument is the instance
            if not hasattr(self, 'fit_time_'):
                self.fit_time_ = []
            self.fit_time_.append(duration)
            
            print(f"{name} imputation completed in {duration:.2f} seconds")
            return result
        return wrapper
    return decorator