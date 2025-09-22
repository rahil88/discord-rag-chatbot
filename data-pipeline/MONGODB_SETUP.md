# MongoDB Atlas Integration Setup Guide

This guide explains how to set up and use MongoDB Atlas integration with the Discord RAG Chatbot document ingestion pipeline.

## Overview

The document ingestion pipeline now includes MongoDB Atlas integration that allows you to:
- Store document chunks with embeddings in MongoDB
- Create indexes for efficient querying
- Support vector similarity search (with Atlas Search)
- Scale your RAG system with a cloud database

## Prerequisites

1. **MongoDB Atlas Account**: Sign up at [MongoDB Atlas](https://www.mongodb.com/atlas)
2. **Python Dependencies**: Install required packages (already included in requirements.txt)
3. **Environment Configuration**: Set up your MongoDB connection string

## Setup Instructions

### 1. Create MongoDB Atlas Cluster

1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas) and sign up/login
2. Create a new cluster (free tier available)
3. Create a database user with read/write permissions
4. Whitelist your IP address (or use 0.0.0.0/0 for development)

### 2. Get Connection String

1. In your Atlas dashboard, click "Connect" on your cluster
2. Choose "Connect your application"
3. Copy the connection string (it looks like):
   ```
   mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
   ```

### 3. Configure Environment Variables

Create a `.env` file in the `data-pipeline` directory:

```bash
# MongoDB Atlas Configuration
MONGODB_CONNECTION_STRING=mongodb+srv://your-username:your-password@your-cluster.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=discord_rag
MONGODB_COLLECTION=document_chunks

# Optional: Disable MongoDB integration
# ENABLE_MONGODB=false

# Optional: Disable embedding generation
# ENABLE_EMBEDDINGS=false

# Processing configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TIMEOUT_SECONDS=30
OUTPUT_DIR=source_documents
ENABLE_CHECKSUMS=true
```

**Important**: Replace `your-username`, `your-password`, and `your-cluster` with your actual MongoDB Atlas credentials.

### 4. Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (including pymongo)
cd data-pipeline
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the ingestion pipeline with MongoDB integration:

```bash
# With MongoDB integration (default)
python ingest.py

# Without MongoDB integration
python ingest.py --no-mongodb

# With custom configuration
python ingest.py --chunk-size 1500 --chunk-overlap 300
```

### Test MongoDB Integration

Use the test script to verify your MongoDB setup:

```bash
python test_mongodb_integration.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_CONNECTION_STRING` | None | MongoDB Atlas connection string |
| `MONGODB_DATABASE` | `discord_rag` | Database name |
| `MONGODB_COLLECTION` | `document_chunks` | Collection name |
| `ENABLE_MONGODB` | `true` | Enable/disable MongoDB integration |
| `ENABLE_EMBEDDINGS` | `true` | Enable/disable embedding generation |

## Database Schema

The MongoDB collection stores documents with the following structure:

```json
{
  "_id": "ObjectId",
  "content": "Document chunk text content",
  "metadata": {
    "source": "Original document URL",
    "title": "Document title",
    "document_id": 0,
    "chunk_id": 0,
    "total_chunks": 12,
    "chunk_size": 1000,
    "chunk_hash": "sha256_hash",
    "embedding": [0.1, 0.2, ...],
    "created_at": "2025-09-20 13:35:27",
    "language": "en",
    "chunk_type": "text"
  },
  "document_title": "AI_Bootcamp_Journey_Learning_Path",
  "chunk_id": 0,
  "total_chunks": 12,
  "chunk_size": 1000,
  "chunk_hash": "sha256_hash",
  "embedding": [0.1, 0.2, ...],
  "created_at": "2025-09-20 13:35:27",
  "source_url": "https://docs.google.com/document/d/...",
  "language": "en",
  "chunk_type": "text"
}
```

## Indexes

The system automatically creates the following indexes for optimal performance:

1. **document_title**: For filtering by document
2. **chunk_id**: For chunk lookups
3. **content (text)**: For full-text search
4. **embedding**: For vector similarity search (requires Atlas Search)

## Vector Search (Optional)

For vector similarity search, you need to configure Atlas Search:

1. In your Atlas dashboard, go to "Search" tab
2. Create a search index with the following configuration:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 768,
        "similarity": "cosine"
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Connection Timeout**: Check your IP whitelist and connection string
2. **Authentication Failed**: Verify username/password in connection string
3. **Import Errors**: Ensure pymongo is installed: `pip install pymongo>=4.6.0`

### Debug Mode

Enable verbose logging for debugging:

```bash
python ingest.py --verbose
```

### Test Connection

Test your MongoDB connection:

```bash
python -c "
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()
client = MongoClient(os.getenv('MONGODB_CONNECTION_STRING'))
client.admin.command('ping')
print('✅ MongoDB connection successful!')
"
```

## Security Best Practices

1. **Use Environment Variables**: Never hardcode credentials
2. **IP Whitelisting**: Restrict access to your IP addresses
3. **Database Users**: Create dedicated users with minimal required permissions
4. **Connection String**: Use SSL/TLS (included in Atlas connection strings)

## Performance Optimization

1. **Batch Inserts**: The system uses batch inserts for efficiency
2. **Indexes**: Automatic index creation for common queries
3. **Connection Pooling**: Built-in connection management
4. **Error Handling**: Graceful fallback if MongoDB is unavailable

## Next Steps

After setting up MongoDB integration:

1. **API Integration**: Update your API server to query MongoDB
2. **Vector Search**: Configure Atlas Search for similarity queries
3. **Monitoring**: Set up MongoDB monitoring and alerts
4. **Backup**: Configure automated backups in Atlas

## Support

For issues with:
- **MongoDB Atlas**: Check [MongoDB Atlas Documentation](https://docs.atlas.mongodb.com/)
- **This Integration**: Check the logs and test scripts
- **General Setup**: Review the main README.md file
