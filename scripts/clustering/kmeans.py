# Copyright 2025 Alex Erofeev / AIGENTTO
# Created by Alex Erofeev at AIGENTTO (http://aigentto.com/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from typing import Dict, Any, Tuple

from .base import Clusterer

class KMeansClusterer(Clusterer):
    """K-Means clustering implementation."""
    
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        """
        Initialize KMeans clusterer.
        
        Args:
            n_clusters: The number of clusters to form
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._kmeans = SKLearnKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10  # Number of time the k-means algorithm will be run with different centroid seeds
        )
    
    def fit_predict(self, embeddings: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit KMeans to the data and return cluster assignments.
        
        Args:
            embeddings: Input data embeddings to cluster
            **kwargs: Additional parameters (ignored for KMeans)
            
        Returns:
            Tuple of (cluster_assignments, metadata) where metadata contains
            information like cluster centers and inertia
        """
        # Determine the actual number of clusters to use (can't be more than samples)
        n_clusters = min(self.n_clusters, len(embeddings))
        if n_clusters < self.n_clusters:
            self._kmeans.n_clusters = n_clusters
        
        # Fit and predict
        cluster_assignments = self._kmeans.fit_predict(embeddings)
        
        # Prepare metadata
        metadata = {
            'n_clusters': n_clusters,
            'inertia': float(self._kmeans.inertia_),
            'cluster_centers': self._kmeans.cluster_centers_.tolist(),
            'algorithm': 'kmeans'
        }
        
        return cluster_assignments, metadata
    
    @property
    def name(self) -> str:
        return "kmeans"
