# Data Flow Analysis - Discord RAG Chatbot

## Current Data Flow

```
Google Docs → Data Pipeline → API Server → Discord Bot
     ↓              ↓            ↓           ↓
  Documents    Chunks +      RAG Service   User Interface
              Embeddings     (Azure AI)
```

## Detailed Flow

### 1. Document Ingestion (data-pipeline/)
```
Google Docs URLs → GoogleDocsDownloader → Raw Content
     ↓
LangChainDocumentChunker → Chunks (1000 chars, 200 overlap)
     ↓
EmbeddingService (BGE-base-en-v1.5) → Vector Embeddings
     ↓
Output Files:
- all_chunks.json (chunks + metadata)
- embeddings.json (vectors)
- individual chunk files
- MongoDB collection (optional)
```

### 2. RAG Processing (api-server/)
```
User Query → RAGService.search_documents()
     ↓
Query Embedding (BGE model) → Similarity Search
     ↓
Top-K Relevant Chunks → Azure AI Foundry (DeepSeek-R1)
     ↓
Enhanced Response with Sources
```

### 3. Discord Integration (discord-bot/)
```
Discord Message → Bot Command → API Call → Response
     ↓
Currently: NOT IMPLEMENTED
```

## Identified Bottlenecks

1. **Discord Bot Missing**: No actual Discord integration
2. **API Server Not Running**: No FastAPI server implementation
3. **MongoDB Underutilized**: Local file search instead of vector DB
4. **No Caching**: Embeddings recomputed on each search
5. **Missing Error Handling**: Discord bot has no error handling

## Performance Metrics

- Document Processing: ~50 chunks in seconds
- Embedding Generation: BGE-base-en-v1.5 (768 dimensions)
- Search Performance: Cosine similarity on local files
- Response Generation: Azure AI Foundry DeepSeek-R1
- Current Latency: Not measured (no running system)

## Optimization Opportunities

1. **Implement FastAPI Server**: Create REST API endpoints
2. **Complete Discord Bot**: Add slash commands and message handling
3. **MongoDB Vector Search**: Use Atlas Search for better performance
4. **Caching Layer**: Cache embeddings and frequent queries
5. **Streaming Responses**: For long responses in Discord
6. **Rate Limiting**: Prevent API abuse
7. **Monitoring**: Add metrics and logging
