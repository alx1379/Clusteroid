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


from typing import Dict, Type, Any, Optional
import numpy as np

from .base import Clusterer
from .kmeans import KMeansClusterer
from .hdbscan import HDBSCANClusterer

class ClustererFactory:
    """Factory class for creating clustering algorithm instances."""
    
    # Registry of available clustering algorithms
    _registry: Dict[str, Type[Clusterer]] = {
        'kmeans': KMeansClusterer,
        'hdbscan': HDBSCANClusterer
    }
    
    @classmethod
    def register_algorithm(cls, name: str, algorithm_class: Type[Clusterer]) -> None:
        """Register a new clustering algorithm.
        
        Args:
            name: Name to register the algorithm under
            algorithm_class: The Clusterer class to register
        """
        if not issubclass(algorithm_class, Clusterer):
            raise ValueError(f"Algorithm class must be a subclass of Clusterer")
        cls._registry[name.lower()] = algorithm_class
    
    @classmethod
    def get_available_algorithms(cls) -> Dict[str, Type[Clusterer]]:
        """Get a dictionary of available clustering algorithms."""
        return cls._registry.copy()
    
    @classmethod
    def create(cls, algorithm_name: str, **kwargs) -> Clusterer:
        """Create an instance of the specified clustering algorithm.
        
        Args:
            algorithm_name: Name of the algorithm to create
            **kwargs: Arguments to pass to the algorithm's constructor
            
        Returns:
            An instance of the specified clustering algorithm
            
        Raises:
            ValueError: If the specified algorithm is not found
        """
        algorithm_name = algorithm_name.lower()
        if algorithm_name not in cls._registry:
            available = ", ".join(f"'{name}'" for name in cls._registry.keys())
            raise ValueError(
                f"Unknown clustering algorithm: '{algorithm_name}'. "
                f"Available algorithms are: {available}"
            )
        
        return cls._registry[algorithm_name](**kwargs)
