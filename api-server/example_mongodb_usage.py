#!/usr/bin/env python3
"""
Example usage of MongoDB vector search in RAG service.

This script demonstrates how to:
1. Convert queries to embeddings using the BGE model
2. Perform vector search against MongoDB collection
3. Retrieve top 3-5 most relevant chunks
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_service import create_mongodb_rag_service

def main():
    """Example usage of MongoDB vector search."""
    
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB connection string
    mongodb_connection_string = os.getenv('MONGODB_CONNECTION_STRING')
    
    if not mongodb_connection_string:
        print("❌ Please set MONGODB_CONNECTION_STRING in your .env file")
        print("Example: mongodb+srv://username:password@cluster.mongodb.net/")
        return
    
    print("🚀 MongoDB Vector Search Example")
    print("=" * 50)
    
    try:
        # Create MongoDB-enabled RAG service
        rag_service = create_mongodb_rag_service(
            connection_string=mongodb_connection_string,
            database="discord_rag",
            collection="document_chunks"
        )
        
        print("✅ Connected to MongoDB successfully!")
        
        # Example queries
        queries = [
            "What is the AI bootcamp curriculum?",
            "How long does the training take?",
            "What are the prerequisites for the program?",
            "Tell me about the internship opportunities",
            "What programming languages will I learn?"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: '{query}'")
            print("-" * 40)
            
            # This function internally:
            # 1. Converts the query to an embedding using the BGE model
            # 2. Performs vector search against MongoDB collection
            # 3. Returns top 3-5 most relevant chunks
            response = rag_service.generate_response(query, top_k=5)
            
            print(f"📝 Answer: {response.answer}")
            print(f"📊 Confidence: {response.confidence:.2f}")
            print(f"📚 Found {len(response.sources)} relevant chunks")
            print(f"⏱️  Search time: {response.processing_time:.3f}s")
            
            # Show the sources
            if response.sources:
                print("📖 Sources:")
                for i, source in enumerate(response.sources, 1):
                    print(f"   {i}. {source.source_document} (similarity: {source.similarity_score:.3f})")
                    print(f"      Content: {source.content[:100]}...")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   1. MongoDB Atlas is set up with vector search index")
        print("   2. Documents are ingested with embeddings")
        print("   3. Connection string is correct")

if __name__ == "__main__":
    main()
