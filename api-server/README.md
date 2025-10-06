# Simple RAG Chatbot API

A minimal FastAPI backend server for RAG (Retrieval-Augmented Generation) chatbot functionality with a single endpoint.

## 🚀 Features

- **Single Endpoint**: Simple `POST /api/ask` for asking questions
- **RAG Integration**: Retrieval-Augmented Generation for intelligent responses
- **MongoDB Integration**: Document storage and retrieval
- **Auto-generated Documentation**: Interactive API docs at `/docs`
- **CORS Support**: Cross-origin resource sharing configuration
- **Environment Configuration**: Flexible configuration management

## 📁 Project Structure

```
api-server/
├── src/
│   ├── main.py              # Main FastAPI application
│   ├── config.py            # Configuration management
│   ├── rag_service.py       # RAG service implementation
│   └── azure_ai_client.py   # Azure AI client
├── requirements.txt         # Python dependencies
├── env.example             # Environment variables template
├── start_server.py         # Server startup script
└── test_server.py          # Test script
```

## 🛠️ Installation & Setup

### 1. Prerequisites

- Python 3.8+
- MongoDB (local or Atlas)
- Virtual environment (recommended)

### 2. Environment Setup

```bash
# Clone the repository
cd discord-rag-chatbot/api-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Copy environment template
cp env.example .env

# Edit .env file with your configuration
nano .env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Edit the `.env` file with your specific configuration:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
ENVIRONMENT=development

# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=discord_rag_chatbot

# AI Configuration (choose one)
OPENAI_API_KEY=your-openai-api-key
# OR
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-azure-openai-key
AZURE_OPENAI_DEPLOYMENT=your-deployment-name

# RAG Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SIMILARITY_THRESHOLD=0.7
MAX_RESULTS=5
```

## 🚀 Running the Server

### Option 1: Using the Startup Script

```bash
python start_server.py
```

### Option 2: Direct Uvicorn Command

```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📚 API Documentation

Once the server is running, you can access:

- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🔗 API Endpoints

### Ask Question
- `POST /api/ask` - Ask a question and get an AI-generated answer using RAG

**Request Body:**
```json
{
  "query": "What is machine learning?"
}
```

**Response:**
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "sources": [
    {
      "title": "Introduction to ML",
      "content": "Machine learning involves...",
      "similarity": 0.85
    }
  ]
}
```

### Root Endpoint
- `GET /` - Basic server information

## 🧪 Testing the API

### Using curl

```bash
curl -X POST "http://localhost:8000/api/ask" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is machine learning?"
     }'
```

### Using the test script

```bash
python test_server.py
```

## 🐳 Docker Support

Build and run with Docker:

```bash
# Build the image
docker build -t rag-chatbot-api .

# Run the container
docker run -p 8000:8000 --env-file .env rag-chatbot-api
```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**: Change the port in `.env` or kill the process using port 8000
2. **MongoDB connection failed**: Check MongoDB URL and ensure MongoDB is running
3. **Missing dependencies**: Run `pip install -r requirements.txt`
4. **Environment variables not loaded**: Ensure `.env` file exists and is properly formatted

### Logs

The server provides detailed logging. Check the console output for error messages and debugging information.

## 📝 License

This project is part of the Discord RAG Chatbot system. See the main project README for license information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the API documentation at `/docs`
- Review the logs for error messages
- Ensure all environment variables are properly configured
- Verify MongoDB and AI service connections