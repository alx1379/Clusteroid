import chromadb
from pathlib import Path

def inspect_chromadb():
    # Set up the ChromaDB client
    db_path = Path(__file__).parent.parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))
    
    # List all collections
    collection_names = client.list_collections()
    print("\n=== Collections in ChromaDB ===")
    for name in collection_names:
        print(f"- {name}")
    
    # Inspect the chunks collection
    if "chunks" in collection_names:
        chunks = client.get_collection("chunks")
        print("\n=== Chunks Collection ===")
        print(f"Total chunks: {chunks.count()}")
        
        # Get first few chunks to inspect their structure
        sample = chunks.get(limit=3)
        if sample and 'ids' in sample and sample['ids']:
            print("\nSample chunk metadata:")
            for i, doc_id in enumerate(sample['ids']):
                metadata = sample['metadatas'][i] if 'metadatas' in sample and i < len(sample['metadatas']) else {}
                document = sample['documents'][i] if 'documents' in sample and i < len(sample['documents']) else ""
                print(f"\nChunk {i+1} (ID: {doc_id}):")
                print(f"  Metadata: {metadata}")
                print(f"  Content preview: {document[:200]}...")
    
    # Inspect the cluster summaries
    if "cluster_summaries" in collection_names:
        summaries = client.get_collection("cluster_summaries")
        print("\n=== Cluster Summaries ===")
        print(f"Total summaries: {summaries.count()}")
        
        # Get all summaries
        all_summaries = summaries.get()
        if all_summaries and 'ids' in all_summaries and all_summaries['ids']:
            for i, doc_id in enumerate(all_summaries['ids']):
                metadata = all_summaries['metadatas'][i] if 'metadatas' in all_summaries and i < len(all_summaries['metadatas']) else {}
                document = all_summaries['documents'][i] if 'documents' in all_summaries and i < len(all_summaries['documents']) else ""
                print(f"\nSummary {i+1} (ID: {doc_id}):")
                print(f"  Cluster ID: {metadata.get('cluster_id', 'N/A')}")
                print(f"  Summary: {document[:200]}...")
                print(f"  Metadata: {metadata}")

if __name__ == "__main__":
    inspect_chromadb()
