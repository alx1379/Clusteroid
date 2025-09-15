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
    
    def get_relevant_clusters(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the most relevant clusters for a query."""
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Query cluster summaries
        results = self.summaries_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.summaries_collection.count())
        )
        
        # Process results
        clusters = []
        for i in range(len(results['ids'][0])):
            cluster_id = int(results['metadatas'][0][i]['cluster_id'])
            distance = 1 - results['distances'][0][i]  # Convert to similarity score
            
            clusters.append({
                'cluster_id': cluster_id,
                'similarity': float(distance),
                'summary': results['documents'][0][i],
                'num_chunks': results['metadatas'][0][i]['num_chunks']
            })
        
        return clusters
    
    def retrieve_from_clusters(self, query: str, cluster_ids: List[int], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from specific clusters."""
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Build filter for the specified clusters
        cluster_filters = [{"cluster_id": cid} for cid in cluster_ids]
        
        # Query chunks from the specified clusters
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where={"$or": cluster_filters}
        )
        
        # Process results
        chunks = []
        for i in range(len(results['ids'][0])):
            chunks.append({
                'text': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'title': results['metadatas'][0][i]['title'],
                'chunk_index': results['metadatas'][0][i]['chunk_index'],
                'cluster_id': results['metadatas'][0][i]['cluster_id'],
                'similarity': 1 - results['distances'][0][i]  # Convert to similarity score
            })
        
        return chunks
    
    def query(self, question: str, top_k_clusters: int = 3, top_k_chunks: int = 5) -> Dict[str, Any]:
        """Process a query through the full RAG pipeline."""
        # Step 1: Find relevant clusters
        clusters = self.get_relevant_clusters(question, top_k=top_k_clusters)
        
        if not clusters:
            return {
                'question': question,
                'clusters': [],
                'results': [],
                'message': 'No relevant clusters found.'
            }
        
        # Step 2: Retrieve chunks from the top clusters
        cluster_ids = [c['cluster_id'] for c in clusters]
        chunks = self.retrieve_from_clusters(
            question, 
            cluster_ids=cluster_ids,
            top_k=top_k_chunks
        )
        
        return {
            'question': question,
            'clusters': clusters,
            'results': chunks,
            'message': 'Success'
        }

def print_results(results: Dict[str, Any]) -> None:
    """Print the results in a user-friendly format."""
    print("\n" + "="*80)
    print(f"QUERY: {results['question']}")
    print("="*80)
    
    if not results['clusters']:
        print("\nNo relevant clusters found.")
        return
    
    # Print cluster information
    print("\nRELEVANT CLUSTERS:")
    print("-" * 60)
    for i, cluster in enumerate(results['clusters'], 1):
        print(f"{i}. Cluster {cluster['cluster_id']} (Similarity: {cluster['similarity']:.2f})")
        print(f"   Summary: {cluster['summary']}")
        print(f"   Chunks in cluster: {cluster['num_chunks']}")
        print("-" * 60)
    
    # Print retrieved chunks
    print("\nRETRIEVED CHUNKS:")
    print("-" * 60)
    for i, chunk in enumerate(results['results'], 1):
        print(f"{i}. [{chunk['source']}] {chunk['title']} (Cluster: {chunk['cluster_id']}, Similarity: {chunk['similarity']:.2f})")
        print(f"   {chunk['text']}")
        print("-" * 60)

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
