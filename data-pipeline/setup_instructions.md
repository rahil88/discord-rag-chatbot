# Document Ingestion Setup Instructions

## Overview
This script downloads Google Docs documents and implements document chunking using LangChain for your Discord RAG chatbot.

## Setup Steps

### 1. Install Dependencies
```bash
cd data-pipeline
pip install -r requirements.txt
```

### 2. Google Docs API Setup (Optional - for better access)
If you want to use the Google Docs API for more reliable access:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the Google Docs API
4. Create credentials (OAuth 2.0 Client ID)
5. Download the credentials file and save as `credentials.json` in the data-pipeline directory

**Note**: The script will work without Google API credentials using a fallback method.

### 3. Run the Script
```bash
python ingest.py
```

## What the Script Does

### Document Download
- Downloads content from the 3 Google Docs URLs you provided
- Uses Google Docs API if credentials are available
- Falls back to web scraping if API is not available
- Handles different Google Docs URL formats

### Document Chunking
- Uses LangChain's `RecursiveCharacterTextSplitter`
- Default settings: 1000 characters per chunk, 200 character overlap
- Preserves document structure and context
- Adds metadata to each chunk

### Output Files
The script creates several output files in the `source_documents` directory:

1. **Individual Documents**: `{title}.txt` - Raw document content
2. **Metadata**: `{title}_metadata.json` - Document metadata
3. **All Chunks**: `all_chunks.json` - All chunks in JSON format
4. **Individual Chunks**: `chunks/chunk_XXXX.txt` - Individual chunk files

## Customization

### Chunking Parameters
You can modify the chunking parameters in the `DocumentChunker` class:

```python
self.chunker = DocumentChunker(
    chunk_size=1000,    # Characters per chunk
    chunk_overlap=200   # Overlap between chunks
)
```

### Document URLs
To add or modify documents, update the `document_urls` and `document_titles` lists in the `DocumentIngestionPipeline` class.

## Troubleshooting

### Common Issues

1. **"Google credentials file not found"**
   - This is normal if you haven't set up Google API credentials
   - The script will use the fallback method

2. **"Error downloading document"**
   - Check if the Google Docs URLs are publicly accessible
   - Verify internet connection
   - Some documents may require authentication

3. **Import errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`

### Fallback Method
If the Google Docs API fails, the script automatically tries to download documents using the export URL format. This works for publicly accessible documents.

## Integration with RAG Service

The generated chunks can be easily loaded into your RAG service:

```python
import json

# Load all chunks
with open('source_documents/all_chunks.json', 'r') as f:
    chunks_data = json.load(f)

# Process chunks for your vector database
for chunk in chunks_data:
    content = chunk['content']
    metadata = chunk['metadata']
    # Add to your vector database
```

## Next Steps

1. Run the script to download and chunk your documents
2. Review the generated chunks to ensure quality
3. Adjust chunking parameters if needed
4. Integrate with your RAG service in the `api-server`
5. Test the complete pipeline with your Discord bot
