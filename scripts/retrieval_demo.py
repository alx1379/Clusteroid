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
            # In Chroma v0.6.0, list_collections returns just the names
            collection_names = self.chroma_client.list_collections()
            print(f"Found collections: {collection_names}")
            
            if not collection_names:
                raise ValueError("No collections found in the database.")
                
            if "chunks" not in collection_names or "cluster_summaries" not in collection_names:
                raise ValueError(
                    f"Required collections not found. Found: {collection_names}"
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
            print("\nTroubleshooting steps:")
            print("1. Make sure you've run 'python scripts/index_documents.py'")
            print("2. Check if the database path is correct")
            print("3. Verify that the collections 'chunks' and 'cluster_summaries' exist")
            print(f"\nCurrent working directory: {os.getcwd()}")
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
        print(f"\n=== DEBUG: get_relevant_clusters ===")
        print(f"Query: {query}")
        print(f"Top K: {top_k}")
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Query cluster summaries using the centroid embeddings
        results = self.summaries_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.summaries_collection.count()),
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"DEBUG: Found {len(results.get('ids', [[]])[0])} clusters")
        
        # Process results
        clusters = []
        if results and 'ids' in results and results['ids']:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'] else {}
                cluster_id = int(metadata.get('cluster_id', -1))
                
                # Handle distance/similarity calculation
                similarity = 1.0
                if 'distances' in results and results['distances'] and results['distances'][0]:
                    similarity = 1 - results['distances'][0][i]  # Convert to similarity score
                
                clusters.append({
                    'cluster_id': cluster_id,
                    'similarity': float(similarity),
                    'summary': results['documents'][0][i] if results.get('documents') and results['documents'] else "",
                    'num_chunks': int(metadata.get('num_chunks', 0)),
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
        print(f"\n=== DEBUG: retrieve_from_clusters ===")
        print(f"Query: {query}")
        print(f"Cluster IDs: {cluster_ids}")
        print(f"Top K: {top_k}")
        
        if not cluster_ids:
            print("DEBUG: No cluster IDs provided")
            return []
            
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Get cluster summaries to access centroids and representative chunks
        print("\nDEBUG: Fetching cluster summaries...")
        cluster_summaries = self.summaries_collection.get(
            where={"cluster_id": {"$in": cluster_ids}}
        )
        print(f"DEBUG: Found {len(cluster_summaries['ids'])} cluster summaries")
        
        # Create a mapping of cluster_id to its summary info
        cluster_info = {}
        for i, summary_id in enumerate(cluster_summaries['ids']):
            metadata = cluster_summaries['metadatas'][i]
            cluster_id = int(metadata['cluster_id'])
            cluster_info[cluster_id] = {
                'centroid': metadata.get('centroid'),
                'representative_chunk_id': metadata.get('representative_chunk_id', '')
            }
            print(f"DEBUG: Cluster {cluster_id} - {cluster_summaries['documents'][i][:100]}...")
        
        # Build filter for the specified clusters
        # Ensure we're using the same type as stored in the database (integers)
        where_clause = {"cluster_id": {"$in": cluster_ids}} if cluster_ids else None
        print(f"\nDEBUG: Querying chunks with where clause: {where_clause}")
        print(f"DEBUG: Cluster IDs type: {[type(cid) for cid in cluster_ids]}")
        
        try:
            # Query chunks within the specified clusters
            print("DEBUG: Executing query...")
            results = self.chunks_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"DEBUG: Query results - {len(results.get('ids', [[]])[0])} chunks found")
            if results and 'ids' in results and results['ids']:
                print(f"DEBUG: First chunk ID: {results['ids'][0][0] if results['ids'][0] else 'None'}")
                print(f"DEBUG: First chunk metadata: {results['metadatas'][0][0] if results.get('metadatas') and results['metadatas'] and results['metadatas'][0] else 'None'}")
                print(f"DEBUG: First chunk distance: {results['distances'][0][0] if results.get('distances') and results['distances'] and results['distances'][0] else 'None'}")
            else:
                print("DEBUG: No results returned from query")
                
        except Exception as e:
            print(f"ERROR in query: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
        
        # Process results
        chunks = []
        if results and 'ids' in results and results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'] else {}
                
                # Handle distance/similarity calculation
                similarity = 1.0
                if 'distances' in results and results['distances'] and results['distances'][0]:
                    similarity = 1 - results['distances'][0][i]  # Convert to similarity score
                
                # Get the cluster ID safely
                cluster_id = int(metadata.get('cluster_id', -1))
                
                # Get the cluster info for this chunk
                current_cluster_info = cluster_info.get(cluster_id, {})
        
                # Add chunk to results
                chunks.append({
                    'text': results['documents'][0][i] if results.get('documents') and results['documents'] else "",
                    'similarity': float(similarity),
                    'source': metadata.get('source', ''),
                    'title': metadata.get('title', ''),
                    'chunk_index': int(metadata.get('chunk_index', 0)),
                    'document_id': metadata.get('document_id', ''),
                    'cluster_id': cluster_id,
                    'is_representative': metadata.get('chunk_id') == current_cluster_info.get('representative_chunk_id', '')
                })
        
        # Sort by similarity and return top_k results
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
    
    if not results['results']:
        print("No relevant chunks found in the top clusters.")
        return
        
    for i, chunk in enumerate(results['results'], 1):
        # Safely get chunk fields with defaults
        title = chunk.get('title', 'Untitled')
        source = chunk.get('source', 'Unknown source')
        chunk_idx = chunk.get('chunk_index', 0)
        cluster_id = chunk.get('cluster_id', -1)
        similarity = chunk.get('similarity', 0.0)
        
        # Print chunk header
        print(f"\n{i}. [Score: {similarity:.3f}] {title}")
        print(f"   Source: {source} | Chunk: {chunk_idx} | "
              f"Cluster: {cluster_id}")
        
        # Highlight if this is the representative chunk for its cluster
        if chunk.get('is_representative'):
            print("   ★ REPRESENTATIVE CHUNK FOR THIS CLUSTER")
        
        # Print cluster summary for context if available
        cluster_summary = chunk.get('cluster_summary')
        if cluster_summary:
            print(f"\n   CLUSTER CONTEXT: {cluster_summary[:200]}{'...' if len(str(cluster_summary)) > 200 else ''}")
        
        # Print chunk content if available
        chunk_text = chunk.get('text', '')
        if chunk_text:
            print(f"\n   CHUNK CONTENT: {chunk_text[:300]}{'...' if len(chunk_text) > 300 else ''}")
        else:
            print("\n   [No content available for this chunk]")
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
