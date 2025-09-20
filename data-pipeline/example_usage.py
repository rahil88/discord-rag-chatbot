#!/usr/bin/env python3
"""
Example usage of the updated document ingestion pipeline with embeddings.
"""

import os
import json
from pathlib import Path

# Import the pipeline components
from ingest import DocumentIngestionPipeline, ProcessingConfig

def main():
    """Example of how to use the updated pipeline with embeddings."""
    
    print("🚀 Document Ingestion Pipeline with Embeddings - Example Usage")
    print("=" * 70)
    
    # Configure the pipeline
    config = ProcessingConfig(
        chunk_size=1000,        # Size of each chunk in characters
        chunk_overlap=200,      # Overlap between chunks
        timeout_seconds=30,     # Download timeout
        output_dir="source_documents",  # Output directory
        enable_checksums=True   # Enable checksum generation
    )
    
    # Initialize pipeline with embeddings enabled
    print("📋 Initializing pipeline with BGE-base-en-v1.5 embeddings...")
    pipeline = DocumentIngestionPipeline(
        config=config,
        enable_embeddings=True  # Enable embedding generation
    )
    
    # Process documents
    print("📄 Processing documents...")
    results = pipeline.process_documents()
    
    # Display results
    stats = results['stats']
    print(f"\n📊 Processing Results:")
    print(f"   • Documents processed: {stats['processed_documents']}")
    print(f"   • Total chunks: {stats['total_chunks']}")
    print(f"   • Processing time: {stats['processing_time']:.2f} seconds")
    
    # Check if embeddings were generated
    output_dir = Path(pipeline.config.output_dir)
    embeddings_file = output_dir / "embeddings.json"
    
    if embeddings_file.exists():
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        
        print(f"\n🔍 Embedding Analysis:")
        print(f"   • Total embeddings: {len(embeddings_data)}")
        
        if embeddings_data:
            # Show embedding dimension
            first_embedding = embeddings_data[0]['embedding']
            print(f"   • Embedding dimension: {len(first_embedding)}")
            print(f"   • Model used: BGE-base-en-v1.5")
            
            # Show sample embedding data
            print(f"\n📋 Sample Embedding Data:")
            sample = embeddings_data[0]
            print(f"   • Chunk ID: {sample['chunk_id']}")
            print(f"   • Document: {sample['document_title']}")
            print(f"   • Embedding (first 5 values): {sample['embedding'][:5]}")
    else:
        print("\n⚠️  No embeddings file found. Check if embedding generation was successful.")
    
    # Show output files
    print(f"\n📁 Generated Files:")
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            print(f"   • {file_path.name}: {size:,} bytes")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Use embeddings.json for vector similarity search")
    print(f"   2. Load all_chunks.json for RAG applications")
    print(f"   3. Integrate with your Discord bot for semantic search")
    
    return results

if __name__ == "__main__":
    main()
