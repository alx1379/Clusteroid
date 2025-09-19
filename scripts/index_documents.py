import os
import json
import yaml
import re
import chromadb
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from langchain.schema import Document as LangchainDocument
from dotenv import load_dotenv

# Import custom text splitter
from semantic_splitter import SemanticTextSplitter

# Import clustering components
from clustering.factory import ClustererFactory, Clusterer
from clustering.kmeans import KMeansClusterer
from clustering.hdbscan import HDBSCANClusterer

# Load environment variables
load_dotenv()

class DocumentIndexer:
    def __init__(self, data_dir: str = "data", db_path: str = "chroma_db", 
                 clustering_algorithm: str = 'kmeans', clustering_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the document indexer with data directory and database path.
        
        Args:
            data_dir: Path to data directory (relative to script)
            db_path: Path to store the ChromaDB (relative to script)
            clustering_algorithm: Name of the clustering algorithm to use ('kmeans' or 'hdbscan')
            clustering_params: Dictionary of parameters to pass to the clustering algorithm
        """
        # Default clustering parameters
        self.default_clustering_params = {
            'kmeans': {'n_clusters': 5, 'random_state': 42},
            'hdbscan': {'min_cluster_size': 5, 'min_samples': None, 'metric': 'euclidean'}
        }
        
        # Set up clustering
        self.clustering_algorithm = clustering_algorithm.lower()
        self.clustering_params = clustering_params or self.default_clustering_params.get(self.clustering_algorithm, {})
        
        # Initialize the clusterer
        self.clusterer = self._initialize_clusterer()
        # Get project root directory (one level up from scripts)
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.absolute()
        
        # Resolve paths relative to project root
        self.data_dir = (project_root / data_dir).resolve()
        self.db_path = (project_root / db_path).resolve()
        
        print(f"Project root: {project_root}")
        print(f"Using data directory: {self.data_dir}")
        print(f"Using database path: {self.db_path}")
        
        # Verify data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        # Create database directory if it doesn't exist
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        # Initialize semantic text splitter with paragraph-aware chunking
        self.text_splitter = SemanticTextSplitter(
            target_size=800,
            min_size=400,
            tolerance=200
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        
    def is_gibberish(self, text: str) -> bool:
        """
        Check if the text contains too many non-words or gibberish.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if text contains too many non-words, False otherwise
        """
        if not text.strip():
            return True
            
        # Split into words (preserve both Cyrillic and Latin words)
        words = re.findall(r'\b[\w-]+\b', text)
        if not words:
            return True
            
        # Common word endings in Russian and English
        common_endings = {
            # Russian noun/adj endings
            'ый', 'ий', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие', 'ь', 'ей', 'ом', 'ем', 'ой', 'ей', 
            'ую', 'юю', 'ым', 'им', 'ом', 'ем', 'ых', 'их', 'ыми', 'ими', 'ых', 'их',
            'ам', 'ям', 'ами', 'ями', 'ах', 'ях',
            # English common endings
            'ing', 'tion', 'ment', 'ness', 'ity', 'ance', 'ence', 'ship', 'hood', 'dom', 'ism',
            'er', 'or', 'ist', 'ian', 'ant', 'ent', 'ary', 'ery', 'ory', 'ful', 'less', 'ish',
            'ed', 'en', 'es', 's', 'ly'
        }
        
        # Common word beginnings in Russian and English
        common_beginnings = {
            # Russian prefixes
            'по', 'на', 'за', 'под', 'над', 'пред', 'при', 'про', 'раз', 'рас', 'с', 'со', 'вз', 'вс',
            'вы', 'до', 'из', 'ис', 'низ', 'нис', 'о', 'об', 'обо', 'от', 'ото', 'па', 'пере', 'по',
            'под', 'подъ', 'пра', 'пре', 'пред', 'предъ', 'при', 'про', 'раз', 'разъ', 'с', 'сверх',
            'среди', 'су', 'у', 'чрез', 'через',
            # English prefixes
            'un', 're', 'in', 'im', 'il', 'ir', 'dis', 'en', 'em', 'non', 'in', 'im', 'over', 'mis',
            'sub', 'pre', 'inter', 'fore', 'de', 'trans', 'super', 'semi', 'anti', 'mid', 'under'
        }
        
        def is_valid_word(word: str) -> bool:
            """Check if a word appears to be a valid word in any language."""
            # Very short words are likely valid
            if len(word) <= 2:
                return True
                
            word_lower = word.lower()
            
            # Check for common word patterns
            has_common_ending = any(word_lower.endswith(ending) for ending in common_endings)
            has_common_beginning = any(word_lower.startswith(begin) for begin in common_beginnings)
            
            # Check for mixed case within word (excluding first letter)
            has_mid_caps = any(c.isupper() for c in word[1:]) and not word.isupper()
            
            # Check for mixed scripts (Cyrillic + Latin)
            has_cyrillic = any('а' <= c.lower() <= 'я' or c in 'ёЁ' for c in word)
            has_latin = any('a' <= c.lower() <= 'z' for c in word)
            has_mixed_script = has_cyrillic and has_latin
            
            # Check for repeated characters (like 'aaaa' or 'пррривет')
            has_repeats = any(match.group(0) for match in re.finditer(r'(.)\1{2,}', word))
            
            # A word is valid if:
            # 1. It has a common ending AND beginning, OR
            # 2. It's not mixed script AND has no mid-caps AND no character repeats
            return (
                (has_common_ending and has_common_beginning) or
                (not has_mixed_script and not has_mid_caps and not has_repeats)
            )
        
        # Count valid and invalid words
        valid_words = sum(1 for word in words if is_valid_word(word))
        valid_ratio = valid_words / len(words)
        
        # Consider text as gibberish if less than 70% of words are valid
        return valid_ratio < 0.7
        
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from the data directory."""
        documents = []
        skipped_count = 0
        print(f"Looking for documents in: {self.data_dir.absolute()}")
        
        # List all files in the data directory
        files = list(self.data_dir.glob("*.txt"))
        print(f"Found {len(files)} text files in the data directory.")
        
        for file_path in files:
            print(f"Loading document: {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    if not content.strip():
                        print(f"  Warning: {file_path.name} is empty")
                        skipped_count += 1
                        continue
                        
                    # Check for gibberish/non-words in the text
                    if self.is_gibberish(content):
                        print(f"  Warning: {file_path.name} appears to contain gibberish and will be skipped")
                        skipped_count += 1
                        continue
                        
                    documents.append({
                        'id': file_path.stem,
                        'content': content,
                        'source': str(file_path.name),
                        'title': file_path.stem.replace('_', ' ').title()
                    })
                    print(f"  Successfully loaded {file_path.name} ({len(content)} characters)")
                    
            except Exception as e:
                print(f"  Error loading {file_path.name}: {str(e)}")
                
        print(f"Loaded {len(documents)} documents. Skipped {skipped_count} documents due to being empty or containing broken text.")
        return documents
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into chunks using semantic text splitting.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks with metadata
        """
        chunks = []
        for doc in documents:
            try:
                # Split document into chunks using our semantic splitter
                doc_texts = self.text_splitter.split_text(doc['content'])
                
                for i, text in enumerate(doc_texts):
                    # Skip very small chunks (unless it's the only chunk from the document)
                    if len(text.strip()) < 100 and i > 0 and len(doc_texts) > 1:
                        continue
                        
                    chunks.append({
                        'id': f"{doc['id']}_chunk{i}",
                        'text': text,
                        'source': doc['source'],
                        'title': doc['title'],
                        'chunk_index': i,
                        'document_id': doc['id']
                    })
                    
            except Exception as e:
                print(f"Error processing document {doc.get('id', 'unknown')}: {str(e)}")
                continue
                
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.embedding_model.encode(texts, show_progress_bar=True)
    
    def _initialize_clusterer(self) -> Clusterer:
        """Initialize the clustering algorithm based on configuration."""
        try:
            # For HDBSCAN, we need to handle the metric parameter specially
            if self.clustering_algorithm == 'hdbscan':
                # Ensure we're using a valid metric for HDBSCAN
                metric = self.clustering_params.get('metric', 'euclidean')
                if metric == 'cosine':
                    # For cosine distance, we need to use 'cosine' in HDBSCAN
                    self.clustering_params['metric'] = 'cosine'
                else:
                    # Default to euclidean if not specified or invalid
                    self.clustering_params['metric'] = 'euclidean'
                
                # Remove n_clusters if it exists (not used by HDBSCAN)
                self.clustering_params.pop('n_clusters', None)
            
            return ClustererFactory.create(
                self.clustering_algorithm,
                **self.clustering_params
            )
        except ValueError as e:
            available = ", ".join(f"'{name}'" for name in ClustererFactory.get_available_algorithms().keys())
            raise ValueError(
                f"Failed to initialize clustering algorithm '{self.clustering_algorithm}'. "
                f"Available algorithms are: {available}. Error: {str(e)}"
            )
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit length (L2 norm).
        
        Args:
            embeddings: Input embeddings to normalize
            
        Returns:
            Normalized embeddings with unit length
        """
        if len(embeddings) == 0:
            return embeddings
            
        # Check if embeddings are already normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.allclose(norms, 1.0, rtol=1e-5, atol=1e-8):
            return embeddings
            
        return normalize(embeddings, norm='l2')
    
    def cluster_documents(self, embeddings: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Cluster document chunks using the configured clustering algorithm.
        
        Args:
            embeddings: Document embeddings to cluster
            **kwargs: Additional parameters to pass to the clustering algorithm
            
        Returns:
            Tuple of (cluster_assignments, metadata) where metadata contains
            algorithm-specific information about the clustering
        """
        if len(embeddings) == 0:
            return np.array([]), {}
            
        # Normalize embeddings before clustering
        normalized_embeddings = self._normalize_embeddings(embeddings)
        
        # Only pass relevant parameters to the clusterer based on the algorithm
        if self.clustering_algorithm == 'kmeans' and 'n_clusters' in kwargs:
            return self.clusterer.fit_predict(normalized_embeddings, **kwargs)
        elif self.clustering_algorithm == 'hdbscan':
            # HDBSCAN parameters are set during initialization, so we don't need to pass them here
            return self.clusterer.fit_predict(normalized_embeddings)
        else:
            return self.clusterer.fit_predict(normalized_embeddings, **kwargs)
    
    def generate_cluster_summaries(self, chunks: List[Dict[str, Any]], 
                                 cluster_ids: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate summaries for each cluster using centroid-based approach.
        
        For each cluster:
        1. Calculate the centroid of the cluster
        2. Find the chunk closest to the centroid as the most representative
        3. Use that chunk as the summary
        
        Args:
            chunks: List of document chunks with 'text' and 'embedding' keys
            cluster_ids: Array of cluster assignments for each chunk
            
        Returns:
            List of cluster summaries with 'cluster_id', 'summary', and 'num_chunks'
        """
        unique_clusters = set(cluster_ids)
        summaries = []
        
        # Generate embeddings for all chunks if not already present
        for chunk in chunks:
            if 'embedding' not in chunk:
                chunk['embedding'] = self.embedding_model.encode(chunk['text'])
        
        for cluster_id in unique_clusters:
            # Get chunks and their embeddings for this cluster
            cluster_data = [(chunk, cid, np.array(chunk.get('embedding', []))) 
                          for chunk, cid in zip(chunks, cluster_ids) 
                          if cid == cluster_id]
            
            if not cluster_data:
                continue
                
            cluster_chunks, _, cluster_embeddings = zip(*cluster_data)
            cluster_embeddings = np.array(cluster_embeddings)
            
            # Calculate cluster centroid
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find the chunk whose embedding is closest to the centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            most_representative_idx = np.argmin(distances)
            representative_chunk = cluster_chunks[most_representative_idx]
            
            # Use the most representative chunk as the summary
            # In a production environment, you might want to use an LLM
            # to generate a more concise summary from the representative chunk
            summary_text = representative_chunk['text']
            
            summaries.append({
                'cluster_id': int(cluster_id),
                'summary': summary_text[:2000],  # Limit summary length
                'num_chunks': len(cluster_chunks),
                'centroid': centroid.tolist(),  # Store centroid for future reference
                'representative_chunk_id': representative_chunk.get('id', '')  # Store reference to the chunk
            })
            
        return summaries
    
    def save_to_chroma(self, chunks: List[Dict[str, Any]], 
                      cluster_ids: np.ndarray,
                      cluster_summaries: List[Dict[str, Any]]) -> None:
        """Save chunks and cluster summaries to ChromaDB."""
        # Create or get collections
        try:
            chunks_collection = self.chroma_client.get_collection("chunks")
            summaries_collection = self.chroma_client.get_collection("cluster_summaries")
        except:
            chunks_collection = self.chroma_client.create_collection("chunks")
            summaries_collection = self.chroma_client.create_collection("cluster_summaries")
        
        # Add chunks to collection
        chunk_embeddings = self.generate_embeddings([chunk['text'] for chunk in chunks])
        
        chunks_collection.add(
            embeddings=[embedding.tolist() for embedding in chunk_embeddings],
            documents=[chunk['text'] for chunk in chunks],
            metadatas=[{
                'source': chunk['source'],
                'title': chunk['title'],
                'chunk_index': chunk['chunk_index'],
                'document_id': chunk['document_id'],
                'cluster_id': int(cluster_id)
            } for chunk, cluster_id in zip(chunks, cluster_ids)],
            ids=[chunk['id'] for chunk in chunks]
        )
        
        # Add cluster summaries to collection
        summary_embeddings = self.generate_embeddings(
            [summary['summary'] for summary in cluster_summaries])
        
        summaries_collection.add(
            embeddings=[embedding.tolist() for embedding in summary_embeddings],
            documents=[summary['summary'] for summary in cluster_summaries],
            metadatas=[{
                'cluster_id': summary['cluster_id'],
                'num_chunks': summary['num_chunks']
            } for summary in cluster_summaries],
            ids=[f"cluster_{summary['cluster_id']}" for summary in cluster_summaries]
        )
    
    def cleanup_chromadb(self):
        """
        Delete all collections in ChromaDB to completely reset the database.
        """
        try:
            # Получаем список всех коллекций
            collections = self.chroma_client.list_collections()
            
            # Удаляем каждую коллекцию
            for coll in collections:
                self.chroma_client.delete_collection(name=coll.name)
            
            print("All ChromaDB collections have been deleted.")
            return True
        
        except Exception as e:
            print(f"Error cleaning up ChromaDB: {str(e)}")
            return False

    def run(self, n_clusters: int = 5) -> List[Dict[str, Any]]:
        """
        Run the full document indexing pipeline.
        
        Args:
            n_clusters: Number of clusters to create (for KMeans)
            
        Returns:
            List of cluster summaries
        """
        # Clean up ChromaDB before starting
        self.cleanup_chromadb()
        
        print("Loading documents...")
        documents = self.load_documents()
        
        print(f"Processing {len(documents)} documents...")
        chunks = self.chunk_documents(documents)
        print(f"Split into {len(chunks)} chunks.")
        
        print("Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.generate_embeddings(chunk_texts)
        
        print(f"Clustering documents using {self.clustering_algorithm}...")
        
        # Only pass n_clusters if it's not None (for KMeans)
        cluster_kwargs = {}
        if n_clusters is not None and self.clustering_algorithm == 'kmeans':
            cluster_kwargs['n_clusters'] = n_clusters
            
        cluster_ids, cluster_metadata = self.cluster_documents(chunk_embeddings, **cluster_kwargs)
        
        # Count actual clusters (excluding noise points which are -1)
        unique_clusters = set(cluster_ids)
        num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        
        print(f"Found {num_clusters} clusters")
        if -1 in cluster_ids:  # HDBSCAN might have noise points
            print(f"  - Noise points: {np.sum(cluster_ids == -1)} (assigned to cluster -1)")
        
        print("Generating cluster summaries...")
        cluster_summaries = self.generate_cluster_summaries(chunks, cluster_ids)
        
        # Add clustering metadata to the first cluster summary for reference
        if cluster_summaries and cluster_metadata:
            cluster_summaries[0]['clustering_metadata'] = cluster_metadata
        
        print("Saving to ChromaDB...")
        self.save_to_chroma(chunks, cluster_ids, cluster_summaries)
        
        print(f"Done! Indexed {len(chunks)} chunks across {len(set(cluster_ids))} clusters.")
        return cluster_summaries

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Index documents into ChromaDB with clustering.')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing documents to index (relative to script)')
    parser.add_argument('--db-path', type=str, default='chroma_db',
                       help='Path to store the ChromaDB (relative to script)')
    # Clustering arguments
    clustering_group = parser.add_argument_group('clustering options')
    clustering_group.add_argument('--clustering-algorithm', type=str, default='kmeans',
                                choices=['kmeans', 'hdbscan'],
                                help='Clustering algorithm to use (default: kmeans)')
    
    # KMeans specific args
    kmeans_group = parser.add_argument_group('KMeans options')
    kmeans_group.add_argument('--n-clusters', type=int, default=5,
                            help='Number of clusters to create (only for KMeans, default: 5)')
    
    # HDBSCAN specific args
    hdbscan_group = parser.add_argument_group('HDBSCAN options')
    hdbscan_group.add_argument('--min-cluster-size', type=int, default=5,
                             help='Minimum cluster size (only for HDBSCAN, default: 5)')
    hdbscan_group.add_argument('--min-samples', type=int, default=None,
                             help='Minimum samples in neighborhood (only for HDBSCAN, default: same as min_cluster_size)')
    hdbscan_group.add_argument('--cluster-selection-epsilon', type=float, default=0.0,
                             help='Distance threshold for cluster merging (only for HDBSCAN, default: 0.0)')
    hdbscan_group.add_argument('--metric', type=str, default='euclidean',
                             choices=['euclidean', 'cosine'],
                             help='Distance metric to use for HDBSCAN (default: euclidean)')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Prepare clustering parameters based on selected algorithm
    clustering_params = {}
    if args.clustering_algorithm == 'kmeans':
        clustering_params = {
            'n_clusters': args.n_clusters,
            'random_state': 42
        }
    elif args.clustering_algorithm == 'hdbscan':
        clustering_params = {
            'min_cluster_size': args.min_cluster_size,
            'min_samples': args.min_samples or args.min_cluster_size,
            'cluster_selection_epsilon': args.cluster_selection_epsilon,
            'metric': args.metric
        }
    
    # Initialize and run the indexer
    indexer = DocumentIndexer(
        data_dir=args.data_dir, 
        db_path=args.db_path,
        clustering_algorithm=args.clustering_algorithm,
        clustering_params=clustering_params
    )
    
    # For backward compatibility, pass n_clusters if using KMeans
    n_clusters = args.n_clusters if args.clustering_algorithm == 'kmeans' else None
    cluster_summaries = indexer.run(n_clusters=n_clusters)
    
    # Print cluster summaries
    print("\nCluster Summaries:")
    for summary in cluster_summaries:
        print(f"\nCluster {summary['cluster_id']} ({summary['num_chunks']} chunks):")
        print("-" * 50)
        print(summary['summary'][:500] + "..." if len(summary['summary']) > 500 else summary['summary'])
        print("-" * 50)
