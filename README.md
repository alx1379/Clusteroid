# Clusteroid

Clusteroid is a document clustering and retrieval system that supports multiple clustering algorithms for organizing and searching document collections.: Dynamic RAG with Meta-Level Retrieval

Clusteroid is a Python-based RAG (Retrieval-Augmented Generation) system that implements dynamic document clustering and meta-level retrieval through cluster summaries. This system helps in efficiently retrieving relevant information by first identifying the most relevant document clusters before performing detailed retrieval.

## Attribution
This project was created by [Alex Erofeev](http://aigentto.com/) at [AIGENTTO](http://aigentto.com/).

Licensed under the [Apache License 2.0](LICENSE).

## Features

- **Document Clustering**: Automatically groups similar document chunks into clusters
- **Meta-Level Retrieval**: Uses cluster summaries for efficient routing of queries
- **Dynamic Updates**: Easy to add new documents and update the index
- **Interactive CLI**: Simple command-line interface for testing queries
- **Pre-configured Sample Data**: Comes with sample documents for immediate testing

## Project Structure

```
Clusteroid/
├── data/                   # Sample documents for testing
├── scripts/
│   ├── __init__.py
│   ├── index_documents.py  # Script for indexing and clustering documents
│   └── retrieval_demo.py   # Interactive CLI for querying the system
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Clusteroid
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Clustering Algorithms

Clusteroid supports multiple clustering algorithms:

### K-Means
- **Description**: Partitions documents into k clusters where each document belongs to the cluster with the nearest mean.
- **Use Case**: When you know the number of clusters in advance and want equally sized clusters.
- **Parameters**:
  - `n_clusters`: Number of clusters to form
  - `random_state`: Random seed for reproducibility

### HDBSCAN
- **Description**: Hierarchical Density-Based Spatial Clustering of Applications with Noise. Finds clusters of varying shapes and sizes, and identifies noise points.
- **Use Case**: When you don't know the number of clusters in advance and want to find clusters of varying densities.
- **Parameters**:
  - `min_cluster_size`: Minimum number of points in a cluster
  - `min_samples`: Number of samples in a neighborhood for a point to be considered a core point
  - `cluster_selection_epsilon`: Distance threshold for cluster merging
  - `metric`: Distance metric to use ('euclidean' or 'cosine')

## Usage

### Indexing Documents

```bash
# Using K-Means (default)
python scripts/index_documents.py --data-dir data --n-clusters 5

# Using HDBSCAN with euclidean distance (default)
python scripts/index_documents.py --data-dir data --clustering-algorithm hdbscan --min-cluster-size 5

# Using HDBSCAN with cosine distance
python scripts/index_documents.py --data-dir data --clustering-algorithm hdbscan --min-cluster-size 5 --metric cosine
```

### Available Arguments

```
--data-dir DIR           Directory containing documents to index (default: data)
--db-path PATH           Path to store the ChromaDB (default: chroma_db)

Clustering options:
  --clustering-algorithm {kmeans,hdbscan}
                        Clustering algorithm to use (default: kmeans)
  --n-clusters N         For KMeans: number of clusters to create (default: 5)
  --min-cluster-size N   For HDBSCAN: minimum cluster size (default: 5)
  --min-samples N        For HDBSCAN: minimum samples in neighborhood
  --cluster-selection-epsilon FLOAT
                        For HDBSCAN: distance threshold for cluster merging (default: 0.0)
  --metric {euclidean,cosine}
                        For HDBSCAN: distance metric to use (default: euclidean)
```

## Quick Start

1. **Set up the environment**:
   ```bash
   # Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Index the sample documents**:
   ```bash
   python scripts/index_documents.py --data-dir data --db-path chroma_db --clusters 5
   ```
   This will:
   - Process all documents in the `data/` directory
   - Create document chunks and generate embeddings
   - Cluster the chunks into the specified number of clusters
   - Generate cluster summaries
   - Save everything to the `chroma_db` directory

3. **Run the interactive demo**:
   ```bash
   python scripts/retrieval_demo.py
   ```
   The demo provides a menu where you can:
   - Enter your own queries
   - Run test queries
   - View cluster information and retrieved documents

## Usage Examples

### Example 1: Querying About Remote Work
```
==============================================================
QUERY: What are the requirements for remote work?
==============================================================

RELEVANT CLUSTERS:
------------------------------------------------------------
1. Cluster 2 (Similarity: 0.87)
   Summary: Remote Work Policy 1. Eligibility All full-time employees who have completed their probationary period are eligible for remote work arrangements. Exceptions may be made on a case-by-case basis with manager approval. 2. Work Hours Employees working remotely are expected to be available during core business hours (10:00 AM to 3:00 PM local time). Flexible start and end times outside of core hours are permitted with manager approval. 3. Communication All remote employees must: - Be available via ...
   Chunks in cluster: 6
------------------------------------------------------------
2. Cluster 3 (Similarity: 0.72)
   Summary: 6. Performance Performance will be evaluated based on deliverables and outcomes rather than hours worked. Regular check-ins with managers are required to ensure alignment with team goals. 7. Office Visits Remote employees are expected to visit the office at least once per quarter for team meetings and collaboration. Travel expenses will be covered by the company. 5. First 90 Days - Take on more responsibility in projects - Complete any remaining training - 90-day performance review - Set goals ...
   Chunks in cluster: 5
------------------------------------------------------------

RETRIEVED CHUNKS:
------------------------------------------------------------
1. [remote_work_policy.txt] Remote Work Policy (Cluster: 2, Similarity: 0.89)
   Remote Work Policy 1. Eligibility All full-time employees who have completed their probationary period are eligible for remote work arrangements. Exceptions may be made on a case-by-case basis with manager approval. 2. Work Hours Employees working remotely are expected to be available during core business hours (10:00 AM to 3:00 PM local time). Flexible start and end times outside of core hours are permitted with manager approval. 3. Communication All remote employees must: - Be available via company chat (Slack/MS Teams) during working hours - Attend all scheduled video meetings with camera on - Respond to urgent requests within 30 minutes - Update their calendar with their working hours and availability 4. Equipment - Company will provide a laptop and necessary peripherals - Employees must have a reliable internet connection (minimum 50 Mbps download/10 Mbps upload) - IT support will be provided remotely during business hours 5. Security - All work must be done on company-issued devices - Use of VPN is mandatory when accessing company resources - No sensitive data should be stored on personal devices - All devices must have endpoint protection software installed 6. Performance Performance will be evaluated based on deliverables and outcomes rather than hours worked. Regular check-ins with managers are required to ensure alignment with team goals. 7. Office Visits Remote employees are expected to visit the office at least once per quarter for team meetings and collaboration. Travel expenses will be covered by the company.
------------------------------------------------------------
```

### Example 2: Querying About Benefits
```
==============================================================
QUERY: What health benefits are offered?
==============================================================

RELEVANT CLUSTERS:
------------------------------------------------------------
1. Cluster 0 (Similarity: 0.92)
   Summary: Employee Benefits Package 1. Health & Wellness - Medical, dental, and vision insurance (90% employer-paid) - Health savings account (HSA) with company contribution - Mental health coverage including therapy sessions - Wellness program with gym reimbursement up to $50/month 2. Time Off - Unlimited PTO (manager approval required) - 10 company holidays per year - 12 weeks paid parental leave - Bereavement leave (5 days) 3. Retirement - 401(k) with 4% company match - Financial planning services - Stock option program for eligible employees 4. Professional Development - $2,000 annual education stipend - Conference attendance opportunities - Internal training programs - Mentorship program 5. Workplace Flexibility - Flexible work hours - Remote work options - Co-working space allowance - Home office stipend ($500) 6. Additional Perks - Free lunch on office days - Monthly team outings - Employee assistance program - Commuter benefits 7. Eligibility - Full-time employees eligible on day 1 - Part-time employees eligible for pro-rated benefits - Waiting periods may apply for certain benefits 8. How to Enroll - New hire enrollment within 30 days of start date - Annual open enrollment in November - Qualifying life events allow mid-year changes
   Chunks in cluster: 5
------------------------------------------------------------

RETRIEVED CHUNKS:
------------------------------------------------------------
1. [benefits.txt] Benefits (Cluster: 0, Similarity: 0.94)
   Employee Benefits Package 1. Health & Wellness - Medical, dental, and vision insurance (90% employer-paid) - Health savings account (HSA) with company contribution - Mental health coverage including therapy sessions - Wellness program with gym reimbursement up to $50/month 2. Time Off - Unlimited PTO (manager approval required) - 10 company holidays per year - 12 weeks paid parental leave - Bereavement leave (5 days) 3. Retirement - 401(k) with 4% company match - Financial planning services - Stock option program for eligible employees 4. Professional Development - $2,000 annual education stipend - Conference attendance opportunities - Internal training programs - Mentorship program 5. Workplace Flexibility - Flexible work hours - Remote work options - Co-working space allowance - Home office stipend ($500) 6. Additional Perks - Free lunch on office days - Monthly team outings - Employee assistance program - Commuter benefits 7. Eligibility - Full-time employees eligible on day 1 - Part-time employees eligible for pro-rated benefits - Waiting periods may apply for certain benefits 8. How to Enroll - New hire enrollment within 30 days of start date - Annual open enrollment in November - Qualifying life events allow mid-year changes
------------------------------------------------------------
```

### Example 3: Querying About Security Policies
```
==============================================================
QUERY: What are the security requirements for company devices?
==============================================================

RELEVANT CLUSTERS:
------------------------------------------------------------
1. Cluster 4 (Similarity: 0.85)
   Summary: 7. Incident Response - Report all security incidents to security@company.com - Follow incident response procedures - Preserve evidence when possible 8. Training - Annual security awareness training required - Phishing simulation exercises quarterly - Specialized training for privileged users
   Chunks in cluster: 1
------------------------------------------------------------
2. Cluster 2 (Similarity: 0.78)
   Summary: Remote Work Policy 1. Eligibility All full-time employees who have completed their probationary period are eligible for remote work arrangements. Exceptions may be made on a case-by-case basis with manager approval. 2. Work Hours Employees working remotely are expected to be available during core business hours (10:00 AM to 3:00 PM local time). Flexible start and end times outside of core hours are permitted with manager approval. 3. Communication All remote employees must: - Be available via ...
   Chunks in cluster: 6
------------------------------------------------------------

RETRIEVED CHUNKS:
------------------------------------------------------------
1. [security_policy.txt] Security Policy (Cluster: 4, Similarity: 0.87)
   Information Security Policy 1. Purpose This policy establishes guidelines for protecting company data and information systems from unauthorized access, use, or disclosure. 2. Scope Applies to all employees, contractors, and third parties with access to company systems or data. 3. Access Control - Use strong, unique passwords (minimum 12 characters) - Enable multi-factor authentication (MFA) on all accounts - Access to sensitive data requires explicit approval - Regular access reviews will be conducted 4. Data Classification - Public: Information approved for public release - Internal: General company information - Confidential: Sensitive business information - Restricted: Highly sensitive data (PII, financial, etc.) 5. Device Security - All devices must have endpoint protection - Automatic screen locking after 15 minutes of inactivity - Regular security updates must be installed - Lost or stolen devices must be reported immediately 6. Network Security - Use company VPN when accessing internal resources - Public Wi-Fi must use VPN - Regular network vulnerability scans - Firewall protection on all networks 7. Incident Response - Report all security incidents to security@company.com - Follow incident response procedures - Preserve evidence when possible 8. Training - Annual security awareness training required - Phishing simulation exercises quarterly - Specialized training for privileged users
------------------------------------------------------------
```

## Adding Custom Queries

You can add your own test queries by modifying the `test_queries` list in `retrieval_demo.py`:

```python
test_queries = [
    "What are the company policies on remote work?",
    "How do I request time off?",
    "What are the security protocols for accessing company data?",
    "How does the performance review process work?",
    "What are the benefits offered by the company?",
    # Add your custom queries here
]

## Adding New Documents

1. Add your text documents to the `data/` directory (`.txt` files)
2. Re-run the indexing script:
   ```bash
   python scripts/index_documents.py --data-dir data --db-path chroma_db --clusters 5
   ```
   
   Note: The `--clusters` parameter can be adjusted based on your document collection size and diversity.

## How It Works

1. **Document Processing**:
   - Documents are split into chunks (300-500 tokens)
   - Each chunk is converted to a vector embedding

2. **Clustering**:
   - Chunks are clustered using K-means
   - The number of clusters is configurable

3. **Summary Generation**:
   - For each cluster, a summary is generated
   - Summaries are also converted to embeddings

4. **Query Processing**:
   - User query is converted to an embedding
   - The system finds the most relevant clusters using summary embeddings
   - Retrieves the most relevant chunks from those clusters
   - Returns the results ranked by relevance

## Customization

### Adjusting Chunking
Modify the `chunk_size` and `chunk_overlap` parameters in `index_documents.py`:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # Adjust chunk size
    chunk_overlap=50,        # Adjust overlap between chunks
    length_function=len,
    add_start_index=True,
)
```

### Changing Embedding Model
Update the model name in both `index_documents.py` and `retrieval_demo.py`:

```python
self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Change model here
```

### Modifying Clustering
Adjust the clustering parameters in `index_documents.py`:

```python
def cluster_documents(self, embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    kmeans = KMeans(
        n_clusters=min(n_clusters, len(embeddings)),
        random_state=42,
        n_init=10  # Number of time the k-means algorithm will be run
    )
    return kmeans.fit_predict(embeddings)
```

## Performance Considerations

- **Indexing Time**: Initial indexing may take several minutes for large document collections
- **Query Speed**: Queries typically return in under a second
- **Memory Usage**: The embedding model is loaded into memory
- **Storage**: Vector database and embeddings are stored on disk in the `chroma_db` directory

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **ChromaDB Errors**:
   - Delete the `chroma_db` directory and re-run the indexing
   - Ensure you have write permissions in the project directory

3. **Out of Memory**:
   - Reduce the number of clusters
   - Decrease the chunk size
   - Use a smaller embedding model

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Uses [ChromaDB](https://www.trychroma.com/) for vector storage and retrieval
- Utilizes [sentence-transformers](https://www.sbert.net/) for text embeddings
- Built with [scikit-learn](https://scikit-learn.org/) for clustering
- Inspired by modern RAG architectures and retrieval techniques
