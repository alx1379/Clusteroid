import numpy as np
import hdbscan as hdbscan_lib
from typing import Dict, Any, Tuple, Optional

from .base import Clusterer

class HDBSCANClusterer(Clusterer):
    """HDBSCAN clustering implementation."""
    
    def __init__(self, 
                 min_cluster_size: int = 5,
                 min_samples: Optional[int] = None,
                 metric: str = 'euclidean',
                 cluster_selection_epsilon: float = 0.0,
                 **kwargs):
        """
        Initialize HDBSCAN clusterer.
        
        Args:
            min_cluster_size: The minimum size of clusters
            min_samples: The number of samples in a neighborhood for a point to be considered a core point
            metric: The metric to use for distance computation
            cluster_selection_epsilon: A distance threshold for cluster splits/merges
            **kwargs: Additional parameters passed to HDBSCAN
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.kwargs = kwargs
        
        self._hdbscan = hdbscan_lib.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=cluster_selection_epsilon,
            **kwargs
        )
    
    def fit_predict(self, embeddings: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit HDBSCAN to the data and return cluster assignments.
        
        Args:
            embeddings: Input data embeddings to cluster
            **kwargs: Additional parameters passed to HDBSCAN
            
        Returns:
            Tuple of (cluster_assignments, metadata) where metadata contains
            information like cluster probabilities and exemplars
        """
        # Fit and predict
        cluster_assignments = self._hdbscan.fit_predict(embeddings, **kwargs)
        
        # Get cluster probabilities (HDBSCAN provides probabilities for each point)
        cluster_probabilities = self._hdbscan.probabilities_
        
        # Get exemplars (points most representative of each cluster)
        exemplars = {}
        if hasattr(self._hdbscan, 'exemplars_'):
            for cluster_id in np.unique(cluster_assignments):
                if cluster_id != -1:  # Skip noise points
                    exemplars[int(cluster_id)] = self._hdbscan.exemplars_[cluster_id].tolist()
        
        # Prepare metadata
        metadata = {
            'n_clusters_found': len(set(cluster_assignments)) - (1 if -1 in cluster_assignments else 0),
            'n_noise_points': int(np.sum(cluster_assignments == -1)),
            'cluster_probabilities': cluster_probabilities.tolist() if cluster_probabilities is not None else None,
            'exemplars': exemplars,
            'algorithm': 'hdbscan'
        }
        
        return cluster_assignments, metadata
    
    @property
    def name(self) -> str:
        return "hdbscan"
