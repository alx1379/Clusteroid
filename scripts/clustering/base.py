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
