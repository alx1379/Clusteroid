from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np

class Clusterer(ABC):
    """Abstract base class for clustering algorithms."""
    
    @abstractmethod
    def fit_predict(self, embeddings: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit the clustering algorithm and return cluster assignments.
        
        Args:
            embeddings: Input data embeddings to cluster
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Tuple of (cluster_assignments, metadata) where metadata is a dict
            containing any additional information about the clustering
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the clustering algorithm."""
        pass
