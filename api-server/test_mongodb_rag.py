#!/usr/bin/env python3
"""
Test script for MongoDB vector search functionality in RAG service.

This script demonstrates how to use the MongoDB-enabled RAG service
to perform vector search queries against a MongoDB collection.
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_service import create_mongodb_rag_service, create_rag_service

def test_mongodb_rag():
    """Test MongoDB vector search functionality."""
    
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB connection string from environment
    mongodb_connection_string = os.getenv('MONGODB_CONNECTION_STRING')
    
    if not mongodb_connection_string:
        print("❌ MONGODB_CONNECTION_STRING not found in environment variables")
        print("Please set MONGODB_CONNECTION_STRING in your .env file")
        return False
    
    try:
        print("🚀 Testing MongoDB RAG Service...")
        print("=" * 50)
        
        # Create MongoDB-enabled RAG service
        rag_service = create_mongodb_rag_service(
            connection_string=mongodb_connection_string,
            database="discord_rag",
            collection="document_chunks"
        )
        
        print("✅ MongoDB RAG Service created successfully!")
        
        # Test queries
        test_queries = [
            "What is the AI bootcamp about?",
            "How long is the training program?",
            "What are the requirements for AI engineers?",
            "Tell me about the internship program",
            "What skills will I learn in the bootcamp?"
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries...")
        print("=" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 30)
            
            try:
                # Generate response using MongoDB vector search
                response = rag_service.generate_response(query, top_k=3)
                
                print(f"✅ Answer: {response.answer[:200]}...")
                print(f"📊 Confidence: {response.confidence:.2f}")
                print(f"📚 Sources: {len(response.sources)}")
                print(f"⏱️  Processing time: {response.processing_time:.3f}s")
                
                # Show top sources
                if response.sources:
                    print("📖 Top sources:")
                    for j, source in enumerate(response.sources[:2], 1):
                        print(f"   {j}. {source.source_document} (score: {source.similarity_score:.3f})")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
        
        print("\n" + "=" * 50)
        print("✅ MongoDB RAG Service test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create MongoDB RAG Service: {e}")
        return False

def test_fallback_rag():
    """Test fallback to local file-based RAG service."""
    
    print("\n🔄 Testing fallback to local RAG service...")
    print("=" * 50)
    
    try:
        # Create local file-based RAG service
        rag_service = create_rag_service()
        
        print("✅ Local RAG Service created successfully!")
        
        # Test a simple query
        query = "What is the AI bootcamp about?"
        print(f"\n🔍 Query: '{query}'")
        
        response = rag_service.generate_response(query, top_k=3)
        
        print(f"✅ Answer: {response.answer[:200]}...")
        print(f"📊 Confidence: {response.confidence:.2f}")
        print(f"📚 Sources: {len(response.sources)}")
        print(f"⏱️  Processing time: {response.processing_time:.3f}s")
        
        print("\n✅ Local RAG Service test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create local RAG Service: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 RAG Service Test Suite")
    print("=" * 50)
    
    # Test MongoDB RAG service
    mongodb_success = test_mongodb_rag()
    
    # Test fallback RAG service
    fallback_success = test_fallback_rag()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   MongoDB RAG Service: {'✅ PASS' if mongodb_success else '❌ FAIL'}")
    print(f"   Local RAG Service: {'✅ PASS' if fallback_success else '❌ FAIL'}")
    
    if mongodb_success or fallback_success:
        print("\n🎉 At least one RAG service is working!")
    else:
        print("\n💥 All RAG services failed!")

if __name__ == "__main__":
    main()
