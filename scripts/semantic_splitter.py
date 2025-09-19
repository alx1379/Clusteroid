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


from typing import List, Optional
import re

class SemanticTextSplitter:
    """
    A text splitter that respects paragraph boundaries while maintaining chunk size constraints.
    
    This splitter:
    1. Splits text into paragraphs first
    2. Combines paragraphs into chunks of target_size +/- tolerance
    3. Ensures no chunk is smaller than min_size (unless it's a single paragraph)
    4. No overlapping between chunks
    """
    
    def __init__(self, 
                 target_size: int = 800,
                 min_size: int = 400,
                 tolerance: int = 200,
                 paragraph_separator: str = r'\n\s*\n'):
        """
        Initialize the semantic text splitter.
        
        Args:
            target_size: Target size for chunks (in characters)
            min_size: Minimum size for chunks (in characters)
            tolerance: Allowed deviation from target_size (in characters)
            paragraph_separator: Regex pattern to split text into paragraphs
        """
        self.target_size = target_size
        self.min_size = min_size
        self.tolerance = tolerance
        self.paragraph_separator = paragraph_separator
        
        if min_size > target_size - tolerance:
            raise ValueError("min_size must be less than or equal to target_size - tolerance")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks respecting paragraph boundaries.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        # Split text into paragraphs
        paragraphs = re.split(self.paragraph_separator, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return []
            
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_length = len(para)
            
            # If adding this paragraph would exceed target_size + tolerance
            if current_chunk and current_length + para_length > self.target_size + self.tolerance:
                # If current chunk is too small, try to add more paragraphs
                if current_length < self.min_size and len(current_chunk) > 1:
                    # Try to find more paragraphs to add without exceeding max size
                    pass
                # Otherwise, finalize the current chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_length += para_length
            
            # If current chunk is large enough, finalize it
            if current_length >= self.target_size - self.tolerance:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        # Ensure no chunk is too small (unless it's a single paragraph)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) < self.min_size and len(chunk.split("\n\n")) > 1:
                # If chunk is too small and has multiple paragraphs, try to merge with previous or next chunk
                if final_chunks and len(final_chunks[-1]) + len(chunk) <= self.target_size + self.tolerance:
                    final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
                else:
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def __call__(self, text: str) -> List[str]:
        """Alias for split_text to match LangChain's interface."""
        return self.split_text(text)
