import os
import json
import yaml
import chromadb
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangchainDocument
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentIndexer:
    def __init__(self, data_dir: str = "data", db_path: str = "chroma_db"):
        """
        Initialize the document indexer with data directory and database path.
        
        Args:
            data_dir: Path to data directory (relative to script)
            db_path: Path to store the ChromaDB (relative to script)
        """
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
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from the data directory."""
        documents = []
        print(f"Looking for documents in: {self.data_dir.absolute()}")
        
        # List all files in the data directory
        files = list(self.data_dir.glob("*.txt"))
        print(f"Found {len(files)} text files in the data directory.")
        
        for file_path in files:
            print(f"Loading document: {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        print(f"  Warning: {file_path.name} is empty")
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
                
        print(f"Successfully loaded {len(documents)} documents.")
        return documents
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks."""
        chunks = []
        for doc in documents:
            doc_texts = self.text_splitter.split_text(doc['content'])
            for i, text in enumerate(doc_texts):
                chunks.append({
                    'id': f"{doc['id']}_chunk{i}",
                    'text': text,
                    'source': doc['source'],
                    'title': doc['title'],
                    'chunk_index': i,
                    'document_id': doc['id']
                })
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        return self.embedding_model.encode(texts, show_progress_bar=True)
    
    def cluster_documents(self, embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """Cluster document chunks using KMeans."""
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
        return kmeans.fit_predict(embeddings)
    
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
    
    def run(self, n_clusters: int = 5):
        """Run the full document indexing pipeline."""
        print("Loading documents...")
        documents = self.load_documents()
        
        print(f"Processing {len(documents)} documents...")
        chunks = self.chunk_documents(documents)
        print(f"Split into {len(chunks)} chunks.")
        
        print("Generating embeddings...")
        chunk_texts = [chunk['text'] for chunk in chunks]
        chunk_embeddings = self.generate_embeddings(chunk_texts)
        
        print("Clustering documents...")
        cluster_ids = self.cluster_documents(chunk_embeddings, n_clusters=n_clusters)
        
        print("Generating cluster summaries...")
        cluster_summaries = self.generate_cluster_summaries(chunks, cluster_ids)
        
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
    parser.add_argument('--clusters', type=int, default=5,
                       help='Number of clusters to create (default: 5)')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    indexer = DocumentIndexer(data_dir=args.data_dir, db_path=args.db_path)
    cluster_summaries = indexer.run(n_clusters=args.clusters)
    
    # Print cluster summaries
    print("\nCluster Summaries:")
    for summary in cluster_summaries:
        print(f"\nCluster {summary['cluster_id']} ({summary['num_chunks']} chunks):")
        print("-" * 50)
        print(summary['summary'][:500] + "..." if len(summary['summary']) > 500 else summary['summary'])
        print("-" * 50)
