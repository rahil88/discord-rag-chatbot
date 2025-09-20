#!/usr/bin/env python3
"""
Test script for embedding functionality in the document ingestion pipeline.
"""

import sys
import os
import json
from pathlib import Path

# Add the current directory to the path so we can import from ingest.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from ingest import EmbeddingService, ProcessingConfig, LangChainDocumentChunker
    print("✅ Successfully imported embedding components")
except ImportError as e:
    print(f"❌ Failed to import embedding components: {e}")
    print("Make sure to install dependencies: pip install -r requirements.txt")
    sys.exit(1)

def test_embedding_service():
    """Test the EmbeddingService class."""
    print("\n🧪 Testing EmbeddingService...")
    
    try:
        # Initialize embedding service
        embedding_service = EmbeddingService()
        print("✅ EmbeddingService initialized successfully")
        
        # Test single embedding generation
        test_text = "This is a test document chunk for embedding generation."
        embedding = embedding_service.generate_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Generated embedding with dimension: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
        else:
            print("❌ Failed to generate embedding")
            return False
        
        # Test batch embedding generation
        test_texts = [
            "First test chunk for batch processing.",
            "Second test chunk for batch processing.",
            "Third test chunk for batch processing."
        ]
        
        batch_embeddings = embedding_service.generate_embeddings_batch(test_texts)
        
        if batch_embeddings and len(batch_embeddings) == len(test_texts):
            print(f"✅ Generated {len(batch_embeddings)} batch embeddings")
            for i, emb in enumerate(batch_embeddings):
                if emb and len(emb) > 0:
                    print(f"   Embedding {i+1}: dimension {len(emb)}")
                else:
                    print(f"❌ Empty embedding for text {i+1}")
                    return False
        else:
            print("❌ Failed to generate batch embeddings")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ EmbeddingService test failed: {e}")
        return False

def test_chunker_with_embeddings():
    """Test the LangChainDocumentChunker with embeddings."""
    print("\n🧪 Testing LangChainDocumentChunker with embeddings...")
    
    try:
        # Initialize components
        config = ProcessingConfig(chunk_size=500, chunk_overlap=100)
        embedding_service = EmbeddingService()
        chunker = LangChainDocumentChunker(config, embedding_service)
        
        # Test document
        test_content = """
        This is a test document for chunking and embedding generation.
        It contains multiple paragraphs to test the chunking functionality.
        
        The document should be split into multiple chunks based on the chunk size.
        Each chunk should have its own embedding generated using the BGE model.
        
        This is the third paragraph to ensure we have enough content for chunking.
        The chunker should handle this content properly and generate embeddings.
        """
        
        # Test metadata
        metadata = {
            'title': 'Test Document',
            'source': 'test://document',
            'document_id': 0
        }
        
        # Generate chunks with embeddings
        chunks = chunker.chunk(test_content, metadata)
        
        if chunks and len(chunks) > 0:
            print(f"✅ Generated {len(chunks)} chunks")
            
            # Check each chunk for embeddings
            for i, chunk in enumerate(chunks):
                if chunk.metadata.get('embedding') is not None:
                    embedding = chunk.metadata['embedding']
                    print(f"   Chunk {i+1}: {len(chunk.page_content)} chars, embedding dim: {len(embedding)}")
                else:
                    print(f"❌ Chunk {i+1} missing embedding")
                    return False
            
            return True
        else:
            print("❌ No chunks generated")
            return False
            
    except Exception as e:
        print(f"❌ Chunker test failed: {e}")
        return False

def test_embedding_serialization():
    """Test that embeddings can be serialized to JSON."""
    print("\n🧪 Testing embedding JSON serialization...")
    
    try:
        embedding_service = EmbeddingService()
        test_text = "Test text for JSON serialization."
        embedding = embedding_service.generate_embedding(test_text)
        
        # Test JSON serialization
        test_data = {
            'text': test_text,
            'embedding': embedding,
            'metadata': {
                'chunk_id': 0,
                'embedding_dimension': len(embedding)
            }
        }
        
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        
        if parsed_data['embedding'] == embedding:
            print("✅ Embeddings serialize/deserialize correctly")
            return True
        else:
            print("❌ Embedding serialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        return False

def main():
    """Run all embedding tests."""
    print("🚀 Starting Embedding Pipeline Tests")
    print("=" * 50)
    
    tests = [
        test_embedding_service,
        test_chunker_with_embeddings,
        test_embedding_serialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Embedding pipeline is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
