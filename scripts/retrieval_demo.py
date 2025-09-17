import os
import chromadb
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, db_path: str = "chroma_db"):
        """
        Initialize the RAG system with the ChromaDB path.
        
        Args:
            db_path: Path to the ChromaDB (relative to project root)
        """
        # Get project root directory (one level up from scripts)
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.absolute()
        
        # Resolve path relative to project root
        self.db_path = (project_root / db_path).resolve()
        
        print(f"Project root: {project_root}")
        print(f"Using database path: {self.db_path}")
        
        # Ensure the database directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        
        # Check if the collections exist
        try:
            collections = [c.name for c in self.chroma_client.list_collections()]
            print(f"Found collections: {collections}")
            
            if not collections:
                raise ValueError("No collections found in the database.")
                
            if "chunks" not in collections or "cluster_summaries" not in collections:
                raise ValueError(
                    f"Required collections not found. Found: {collections}"
                )
                
            self.chunks_collection = self.chroma_client.get_collection("chunks")
            self.summaries_collection = self.chroma_client.get_collection("cluster_summaries")
            
            # Print debug info
            print(f"Chunks collection count: {self.chunks_collection.count()}")
            print(f"Summaries collection count: {self.summaries_collection.count()}")
            
        except Exception as e:
            print(f"Error accessing collections: {e}")
            print("\nPlease make sure to run 'python scripts/index_documents.py' first.")
            print(f"Database path: {self.db_path.absolute()}")
            raise
    
    def get_relevant_clusters(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find the most relevant clusters for a query using the cluster centroids.
        
        Args:
            query: The search query
            top_k: Maximum number of clusters to return
            
        Returns:
            List of cluster information including id, similarity, summary, and metadata
        """
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Query cluster summaries using the centroid embeddings
        results = self.summaries_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.summaries_collection.count())
        )
        
        # Process results
        clusters = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            cluster_id = int(metadata['cluster_id'])
            distance = 1 - results['distances'][0][i]  # Convert to similarity score
            
            clusters.append({
                'cluster_id': cluster_id,
                'similarity': float(distance),
                'summary': results['documents'][0][i],
                'num_chunks': int(metadata['num_chunks']),
                'representative_chunk_id': metadata.get('representative_chunk_id', '')
            })
        
        return clusters
    
    def retrieve_from_clusters(self, query: str, cluster_ids: List[int], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from specific clusters with enhanced ranking.
        
        Args:
            query: The search query
            cluster_ids: List of cluster IDs to search within
            top_k: Maximum number of chunks to return
            
        Returns:
            List of relevant chunks with metadata and similarity scores
        """
        if not cluster_ids:
            return []
            
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Get cluster summaries to access centroids and representative chunks
        cluster_summaries = self.summaries_collection.get(
            where={"cluster_id": {"$in": cluster_ids}}
        )
        
        # Create a mapping of cluster_id to its summary info
        cluster_info = {}
        for i, summary_id in enumerate(cluster_summaries['ids']):
            metadata = cluster_summaries['metadatas'][i]
            cluster_info[int(metadata['cluster_id'])] = {
                'centroid': metadata.get('centroid'),
                'representative_chunk_id': metadata.get('representative_chunk_id', '')
            }
        
        # Build filter for the specified clusters
        cluster_filters = [{"cluster_id": cid} for cid in cluster_ids]
        
        # Query more chunks than needed to allow for re-ranking
        query_results = self.chunks_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 3,  # Get more results for better re-ranking
            where={"$or": cluster_filters}
        )
        
        # Process and re-rank results
        chunks = []
        for i in range(len(query_results['ids'][0])):
            chunk_metadata = query_results['metadatas'][0][i]
            cluster_id = int(chunk_metadata['cluster_id'])
            
            # Calculate additional features for re-ranking
            is_representative = (chunk_metadata.get('id') == 
                               cluster_info[cluster_id].get('representative_chunk_id', ''))
            
            # Boost score for representative chunks
            similarity = 1 - query_results['distances'][0][i]
            if is_representative:
                similarity = min(similarity * 1.2, 1.0)  # Boost by 20%
            
            chunks.append({
                'id': query_results['ids'][0][i],
                'text': query_results['documents'][0][i],
                'source': chunk_metadata['source'],
                'title': chunk_metadata['title'],
                'chunk_index': chunk_metadata['chunk_index'],
                'cluster_id': cluster_id,
                'similarity': similarity,
                'is_representative': is_representative
            })
        
        # Sort by combined score and take top_k
        chunks.sort(key=lambda x: x['similarity'], reverse=True)
        return chunks[:top_k]
    
    def query(self, question: str, top_k_clusters: int = 2, top_k_chunks: int = 10) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline with enhanced retrieval.
        
        Args:
            question: The user's question
            top_k_clusters: Number of top clusters to consider
            top_k_chunks: Number of chunks to return
            
        Returns:
            Dictionary containing query results with clusters and relevant chunks
        """
        # Step 1: Find relevant clusters using centroid-based search
        clusters = self.get_relevant_clusters(question, top_k=top_k_clusters)
        
        if not clusters:
            return {
                'question': question,
                'clusters': [],
                'results': [],
                'message': 'No relevant clusters found.'
            }
        
        # Step 2: Retrieve and re-rank chunks from the top clusters
        cluster_ids = [c['cluster_id'] for c in clusters]
        chunks = self.retrieve_from_clusters(
            question, 
            cluster_ids=cluster_ids,
            top_k=top_k_chunks
        )
        
        # Add cluster summary to each chunk for context
        cluster_summaries = {c['cluster_id']: c for c in clusters}
        for chunk in chunks:
            chunk['cluster_summary'] = cluster_summaries[chunk['cluster_id']]['summary']
        
        return {
            'question': question,
            'clusters': clusters,
            'results': chunks,
            'message': 'Success',
            'topics_covered': [{
                'cluster_id': c['cluster_id'],
                'summary': c['summary'],
                'num_chunks': c['num_chunks']
            } for c in clusters]
        }

def print_results(results: Dict[str, Any]) -> None:
    """
    Print the retrieval results in a user-friendly format with enhanced cluster information.
    
    Args:
        results: Dictionary containing query results from RAGSystem.query()
    """
    print("\n" + "="*80)
    print(f"QUERY: {results['question']}")
    print("="*80)
    
    if not results['clusters']:
        print("\nNo relevant clusters found.")
        return
    
    # Print cluster information
    print("\nRELEVANT TOPICS FOUND:")
    print("-" * 80)
    for i, cluster in enumerate(results['clusters'], 1):
        print(f"{i}. [Cluster {cluster['cluster_id']}] - {cluster['summary'][:120]}...")
        print(f"   Similarity: {cluster['similarity']:.3f} | Chunks: {cluster['num_chunks']} | "
              f"Representative: {'Yes' if cluster.get('representative_chunk_id') else 'No'}")
    
    # Print top chunks with cluster context
    print("\nTOP CHUNKS (with cluster context):")
    print("-" * 80)
    for i, chunk in enumerate(results['results'], 1):
        print(f"{i}. [Score: {chunk['similarity']:.3f}] {chunk['title']}")
        print(f"   Source: {chunk['source']} | Chunk: {chunk['chunk_index']} | "
              f"Cluster: {chunk['cluster_id']}")
        
        # Highlight if this is the representative chunk for its cluster
        if chunk.get('is_representative'):
            print("   ★ REPRESENTATIVE CHUNK FOR THIS CLUSTER")
        
        # Print cluster summary for context
        print(f"\n   CLUSTER CONTEXT: {chunk.get('cluster_summary', 'No summary available')[:200]}...")
        
        # Print chunk content
        print(f"\n   CHUNK CONTENT: {chunk['text'][:300]}...")
        print("\n" + "-" * 80 + "\n")

def run_demo():
    """Run the interactive demo."""
    # Initialize the RAG system
    try:
        rag = RAGSystem()
        print("RAG system initialized successfully!")
        print("Type 'exit' to quit the demo.")
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you've run 'python scripts/index_documents.py'")
        print("2. Check if the database path is correct")
        print("3. Verify that the collections 'chunks' and 'cluster_summaries' exist")
        print(f"\nCurrent working directory: {os.getcwd()}")
        return
    
    # Predefined test queries
    test_queries = [
        "What are the company policies on remote work?",
        "How do I request time off?",
        "What are the security protocols for accessing company data?",
        "How does the performance review process work?",
        "What are the benefits offered by the company?"
    ]
    
    while True:
        # Show menu
        print("\n" + "="*50)
        print("RAG SYSTEM DEMO")
        print("="*50)
        print("\nChoose an option:")
        print("1. Enter your own query")
        print("2. Run a test query")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            query = input("\nEnter your query: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
                
            results = rag.query(query)
            print_results(results)
            
        elif choice == '2':
            print("\nTest Queries:")
            for i, q in enumerate(test_queries, 1):
                print(f"{i}. {q}")
                
            try:
                q_choice = int(input("\nSelect a query (1-5): ").strip())
                if 1 <= q_choice <= len(test_queries):
                    query = test_queries[q_choice - 1]
                    print(f"\nRunning query: {query}")
                    results = rag.query(query)
                    print_results(results)
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a number.")
                
        elif choice == '3':
            print("Exiting demo. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    run_demo()
