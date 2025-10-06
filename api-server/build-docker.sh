#!/bin/bash

# Docker build script for the API server
# This script builds the Docker image with the correct build context

set -e  # Exit on any error

echo "🐳 Building Docker image for API server..."

# Get the project root directory (parent of api-server)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_SERVER_DIR="$PROJECT_ROOT/api-server"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 API server dir: $API_SERVER_DIR"

# Check if required data files exist
DATA_DIR="$PROJECT_ROOT/data-pipeline/source_documents"
CHUNKS_FILE="$DATA_DIR/all_chunks.json"
EMBEDDINGS_FILE="$DATA_DIR/embeddings.json"

echo "🔍 Checking for required data files..."
if [ ! -f "$CHUNKS_FILE" ]; then
    echo "⚠️  Warning: $CHUNKS_FILE not found"
    echo "   The container will start but may not have document data"
fi

if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "⚠️  Warning: $EMBEDDINGS_FILE not found"
    echo "   The container will start but may not have embeddings data"
fi

# Build the Docker image
echo "🔨 Building Docker image..."
cd "$PROJECT_ROOT"  # Build from project root for proper context

docker build \
    -f api-server/Dockerfile \
    -t discord-rag-api:latest \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VERSION="1.0.0" \
    .

echo "✅ Docker image built successfully!"
echo ""
echo "🚀 To run the container:"
echo "   docker run -p 8000:8000 --env-file api-server/.env discord-rag-api:latest"
echo ""
echo "📖 To view logs:"
echo "   docker logs <container_id>"
echo ""
echo "🔧 To run with interactive shell:"
echo "   docker run -it --env-file api-server/.env discord-rag-api:latest /bin/bash"
