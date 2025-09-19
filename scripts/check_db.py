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
