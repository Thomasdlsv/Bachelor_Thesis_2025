import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any
from .timed_imputer import TimedImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from bgan.utility.bgan_imp import BGAIN
from bn_bgan.bn_bgan_imp import BN_AUG_Imputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer

class ImputationEvaluator:
    """A class to evaluate imputation methods with timing."""
    
    def __init__(self, imputation_methods: Dict[str, Any]):
        """
        Initialize with a dictionary of imputation methods.
        Args:
            imputation_methods: Dict mapping method names to imputer objects
        """
        self.imputers = {
            name: TimedImputer(imputer, name)
            for name, imputer in imputation_methods.items()
        }
        self.timing_results = pd.DataFrame()
        
    def impute_and_time(self, X: pd.DataFrame, method_name: str) -> np.ndarray:
        """Run imputation for a specific method with timing."""
        if method_name not in self.imputers:
            raise ValueError(f"Unknown imputation method: {method_name}")
            
        imputer = self.imputers[method_name]
        result = imputer.impute_all_missing(X)
        
        # Add timing stats to results
        stats = imputer.get_timing_stats()
        if self.timing_results.empty:
            self.timing_results = pd.DataFrame([stats])
        else:
            self.timing_results = pd.concat([
                self.timing_results, 
                pd.DataFrame([stats])
            ], ignore_index=True)
            
        return result
        
    def get_timing_summary(self) -> pd.DataFrame:
        """Get a summary of all timing measurements."""
        if self.timing_results.empty:
            return pd.DataFrame()
            
        return self.timing_results
        
    def save_timing_results(self, filepath: str):
        """Save timing results to a CSV file."""
        if not self.timing_results.empty:
            self.timing_results.to_csv(filepath, index=False)
            print(f"Timing results saved to {filepath}")