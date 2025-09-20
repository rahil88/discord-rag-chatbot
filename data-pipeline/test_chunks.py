#!/usr/bin/env python3
"""
Test script to verify that the document chunks can be loaded properly.
"""

import json
from pathlib import Path

def test_chunk_loading():
    """Test loading and displaying chunks from the generated files."""
    
    chunks_file = Path("source_documents/all_chunks.json")
    
    if not chunks_file.exists():
        print("❌ Error: all_chunks.json not found. Run ingest.py first.")
        return False
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    print(f"✅ Successfully loaded {len(chunks_data)} chunks")
    
    # Display summary statistics
    total_chars = sum(len(chunk['content']) for chunk in chunks_data)
    avg_chunk_size = total_chars / len(chunks_data)
    
    print(f"📊 Statistics:")
    print(f"   - Total chunks: {len(chunks_data)}")
    print(f"   - Total characters: {total_chars:,}")
    print(f"   - Average chunk size: {avg_chunk_size:.0f} characters")
    
    # Show chunk size distribution
    chunk_sizes = [len(chunk['content']) for chunk in chunks_data]
    min_size = min(chunk_sizes)
    max_size = max(chunk_sizes)
    
    print(f"   - Chunk size range: {min_size} - {max_size} characters")
    
    # Show documents processed
    documents = set(chunk['metadata']['title'] for chunk in chunks_data)
    print(f"   - Documents processed: {len(documents)}")
    for doc in sorted(documents):
        doc_chunks = [c for c in chunks_data if c['metadata']['title'] == doc]
        print(f"     • {doc}: {len(doc_chunks)} chunks")
    
    # Display sample chunks
    print(f"\n📄 Sample chunks:")
    for i, chunk in enumerate(chunks_data[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk['metadata']['title']}")
        print(f"Size: {len(chunk['content'])} characters")
        print(f"Content preview: {chunk['content'][:200]}...")
    
    return True

def test_individual_files():
    """Test that individual document files exist and are readable."""
    
    source_dir = Path("source_documents")
    expected_files = [
        "AI_Bootcamp_Journey_Learning_Path.txt",
        "Training_For_AI_Engineer_Interns.txt", 
        "Intern_FAQ_AI_Bootcamp.txt"
    ]
    
    print(f"\n📁 Testing individual files:")
    
    for filename in expected_files:
        file_path = source_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"   ✅ {filename}: {len(content):,} characters")
        else:
            print(f"   ❌ {filename}: Not found")
    
    # Test chunks directory
    chunks_dir = source_dir / "chunks"
    if chunks_dir.exists():
        chunk_files = list(chunks_dir.glob("chunk_*.txt"))
        print(f"   ✅ chunks/ directory: {len(chunk_files)} individual chunk files")
    else:
        print(f"   ❌ chunks/ directory: Not found")

if __name__ == "__main__":
    print("🧪 Testing Document Chunk Loading")
    print("=" * 50)
    
    success = test_chunk_loading()
    test_individual_files()
    
    if success:
        print(f"\n🎉 All tests passed! Your document ingestion pipeline is working correctly.")
        print(f"\nNext steps:")
        print(f"1. Review the generated chunks in source_documents/")
        print(f"2. Adjust chunking parameters in ingest.py if needed")
        print(f"3. Integrate with your RAG service")
    else:
        print(f"\n❌ Tests failed. Please check the error messages above.")
