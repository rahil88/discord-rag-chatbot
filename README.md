# Discord RAG Chatbot

A comprehensive Discord chatbot with Retrieval-Augmented Generation (RAG) capabilities, featuring document ingestion, embedding generation, and intelligent question answering.

## 🚀 Features

- **Document Ingestion Pipeline**: Automated downloading and processing of Google Docs
- **Intelligent Chunking**: Advanced text splitting with configurable parameters
- **Embedding Generation**: BGE-base-en-v1.5 model for semantic similarity
- **RAG Integration**: Vector-based document retrieval for accurate responses
- **Discord Bot**: Real-time chat interface with natural language processing

## 📁 Project Structure

```
discord-rag-chatbot/
├── data-pipeline/          # Document processing and embedding generation
│   ├── ingest.py          # Main ingestion pipeline with embeddings
│   ├── test_embeddings.py # Embedding functionality tests
│   ├── example_usage.py   # Usage examples
│   └── requirements.txt   # Python dependencies
├── api-server/            # FastAPI server for RAG queries
│   ├── main.py           # API endpoints
│   ├── rag_service.py    # RAG implementation
│   └── requirements.txt  # Server dependencies
├── discord-bot/          # Discord bot implementation
│   ├── src/
│   │   ├── index.js     # Bot entry point
│   │   └── commands/    # Bot commands
│   └── package.json     # Node.js dependencies
└── README.md
```

## 🛠️ Setup Instructions

### 1. Data Pipeline Setup

The data pipeline handles document ingestion and embedding generation using the BGE-base-en-v1.5 model.

```bash
cd data-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the ingestion pipeline
python ingest.py

# Test embedding functionality
python test_embeddings.py

# See usage examples
python example_usage.py
```

#### Configuration Options

```bash
# Custom chunk size and overlap
python ingest.py --chunk-size 1500 --chunk-overlap 300

# Disable embeddings (faster processing)
python ingest.py --no-embeddings

# Custom output directory
python ingest.py --output-dir my_documents

# Verbose logging
python ingest.py --verbose
```

#### Environment Variables

```bash
export CHUNK_SIZE=1000
export CHUNK_OVERLAP=200
export ENABLE_EMBEDDINGS=true
export ENABLE_CHECKSUMS=true
export OUTPUT_DIR=source_documents
```

### 2. API Server Setup

```bash
cd api-server

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp env.example .env

# Configure your environment variables
# Edit .env file with your settings

# Run the server
python src/main.py
```

### 3. Discord Bot Setup

```bash
cd discord-bot

# Install dependencies
npm install

# Copy environment template
cp env.example .env

# Configure your Discord bot token
# Edit .env file with your bot token

# Run the bot
npm start
```

## 🔧 Embedding Features

### BGE-base-en-v1.5 Model

The pipeline uses the BGE-base-en-v1.5 embedding model for generating high-quality vector representations:

- **Model**: `BAAI/bge-base-en-v1.5`
- **Dimension**: 768
- **Language**: English
- **Use Case**: Semantic similarity and retrieval

### Generated Files

After running the ingestion pipeline, you'll find:

- `all_chunks.json`: Complete document chunks with metadata
- `embeddings.json`: Vector embeddings for semantic search
- `chunk_index.json`: Quick access index for chunks
- `processing_report.json`: Detailed processing statistics
- `chunks/`: Individual chunk files for easy reading

### Embedding Usage Example

```python
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embeddings
with open('source_documents/embeddings.json', 'r') as f:
    embeddings_data = json.load(f)

# Load the same model for similarity search
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Generate query embedding
query = "What is machine learning?"
query_embedding = model.encode(query)

# Find most similar chunks
similarities = []
for item in embeddings_data:
    chunk_embedding = np.array(item['embedding'])
    similarity = np.dot(query_embedding, chunk_embedding) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
    )
    similarities.append((similarity, item))

# Sort by similarity
similarities.sort(reverse=True, key=lambda x: x[0])

# Get top 3 most similar chunks
top_chunks = similarities[:3]
for similarity, chunk_data in top_chunks:
    print(f"Similarity: {similarity:.3f}")
    print(f"Chunk ID: {chunk_data['chunk_id']}")
    print(f"Document: {chunk_data['document_title']}")
    print()
```

## 📊 Performance

### Processing Statistics

- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters (configurable)
- **Embedding Dimension**: 768
- **Processing Speed**: ~50-100 chunks/minute (depending on hardware)

### Memory Requirements

- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB+
- **GPU**: Optional (CUDA support for faster embedding generation)

## 🔍 Testing

Run the comprehensive test suite:

```bash
cd data-pipeline

# Test embedding functionality
python test_embeddings.py

# Test full pipeline
python ingest.py --test-only

# Validate environment
python ingest.py --validate
```

## 🚀 Usage Examples

### Basic Usage

```python
from ingest import DocumentIngestionPipeline, ProcessingConfig

# Configure pipeline
config = ProcessingConfig(
    chunk_size=1000,
    chunk_overlap=200,
    enable_checksums=True
)

# Initialize with embeddings
pipeline = DocumentIngestionPipeline(
    config=config,
    enable_embeddings=True
)

# Process documents
results = pipeline.process_documents()
print(f"Generated {results['stats']['total_chunks']} chunks with embeddings")
```

### Advanced Configuration

```python
# Custom embedding model
from ingest import EmbeddingService

embedding_service = EmbeddingService("BAAI/bge-large-en-v1.5")
pipeline = DocumentIngestionPipeline(
    config=config,
    enable_embeddings=True
)
pipeline.embedding_service = embedding_service
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:

1. Check the test files for usage examples
2. Review the processing reports for debugging
3. Enable verbose logging with `--verbose` flag
4. Check the logs in `ingestion.log`

## 🔄 Updates

### Version 2.0.0
- ✅ Added BGE-base-en-v1.5 embedding generation
- ✅ Enhanced chunking with embedding metadata
- ✅ Separate embeddings.json file for easy access
- ✅ Comprehensive testing suite
- ✅ CLI options for embedding control
- ✅ Batch embedding processing for efficiency
