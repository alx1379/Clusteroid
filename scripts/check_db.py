import chromadb
from pathlib import Path
import os

def check_database(db_path="../chroma_db"):
    print(f"Checking database at: {os.path.abspath(db_path)}")
    
    # Ensure the database directory exists
    if not os.path.exists(db_path):
        print(f"Database directory does not exist: {db_path}")
        return
    
    # Initialize ChromaDB client
    try:
        client = chromadb.PersistentClient(path=db_path)
        print("Successfully connected to ChromaDB")
        
        # List all collections
        collections = client.list_collections()
        print(f"\nFound {len(collections)} collections:")
        
        for collection in collections:
            print(f"\nCollection: {collection.name}")
            print(f"  ID: {collection.id}")
            print(f"  Metadata: {collection.metadata}")
            try:
                count = collection.count()
                print(f"  Item count: {count}")
                
                # Get a sample item
                if count > 0:
                    sample = collection.get(limit=1)
                    print("  Sample item keys:", list(sample.keys()))
            except Exception as e:
                print(f"  Error getting collection info: {e}")
        
    except Exception as e:
        print(f"Error accessing ChromaDB: {e}")

if __name__ == "__main__":
    check_database()
