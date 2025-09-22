#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) Service for Discord RAG Chatbot.

This module provides intelligent document retrieval and response generation
capabilities using semantic similarity search and embeddings.

Author: AI Assistant
Version: 1.0.0
License: MIT
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result from the RAG system."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    source_document: str
    chunk_id: int

@dataclass
class RAGResponse:
    """Represents a complete RAG response."""
    answer: str
    sources: List[SearchResult]
    confidence: float
    processing_time: float

class RAGService:
    """Main RAG service for document retrieval and response generation."""
    
    def __init__(self, 
                 chunks_file: str = "/Users/craddy-san/Desktop/Projects/discord-rag-chatbot/discord-rag-chatbot/data-pipeline/source_documents/all_chunks.json",
                 embeddings_file: str = "/Users/craddy-san/Desktop/Projects/discord-rag-chatbot/discord-rag-chatbot/data-pipeline/source_documents/embeddings.json",
                 model_name: str = "BAAI/bge-base-en-v1.5",
                 mongodb_connection_string: Optional[str] = None,
                 mongodb_database: str = "discord_rag",
                 mongodb_collection: str = "document_chunks",
                 use_mongodb: bool = False):
        """
        Initialize the RAG service.
        
        Args:
            chunks_file: Path to the chunks JSON file
            embeddings_file: Path to the embeddings JSON file
            model_name: Name of the sentence transformer model to use
            mongodb_connection_string: MongoDB connection string for vector search
            mongodb_database: MongoDB database name
            mongodb_collection: MongoDB collection name
            use_mongodb: Whether to use MongoDB for vector search instead of local files
        """
        self.chunks_file = chunks_file
        self.embeddings_file = embeddings_file
        self.model_name = model_name
        self.model = None
        self.chunks_data = []
        self.embeddings_data = []
        self.document_embeddings = None
        
        # MongoDB configuration
        self.use_mongodb = use_mongodb
        self.mongodb_connection_string = mongodb_connection_string
        self.mongodb_database = mongodb_database
        self.mongodb_collection = mongodb_collection
        self.mongodb_client = None
        self.mongodb_collection_obj = None
        
        # Initialize the service
        self._load_model()
        if self.use_mongodb and self.mongodb_connection_string:
            self._connect_mongodb()
        else:
            self._load_data()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _connect_mongodb(self) -> None:
        """Connect to MongoDB for vector search."""
        try:
            logger.info("Connecting to MongoDB for vector search...")
            self.mongodb_client = MongoClient(
                self.mongodb_connection_string,
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            
            # Test the connection
            self.mongodb_client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            # Initialize collection
            database = self.mongodb_client[self.mongodb_database]
            self.mongodb_collection_obj = database[self.mongodb_collection]
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.warning("Falling back to local file-based search")
            self.use_mongodb = False
            self._load_data()
        except Exception as e:
            logger.error(f"Unexpected MongoDB connection error: {e}")
            logger.warning("Falling back to local file-based search")
            self.use_mongodb = False
            self._load_data()
    
    def _load_data(self) -> None:
        """Load chunks and embeddings data from files."""
        try:
            # Load chunks data
            if os.path.exists(self.chunks_file):
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    self.chunks_data = json.load(f)
                logger.info(f"Loaded {len(self.chunks_data)} chunks from {self.chunks_file}")
            else:
                logger.warning(f"Chunks file not found: {self.chunks_file}")
                self.chunks_data = []
            
            # Load embeddings data
            if os.path.exists(self.embeddings_file):
                with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                    self.embeddings_data = json.load(f)
                logger.info(f"Loaded {len(self.embeddings_data)} embeddings from {self.embeddings_file}")
                
                # Prepare embeddings matrix for similarity search
                self._prepare_embeddings_matrix()
            else:
                logger.warning(f"Embeddings file not found: {self.embeddings_file}")
                self.embeddings_data = []
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _prepare_embeddings_matrix(self) -> None:
        """Prepare embeddings matrix for efficient similarity search."""
        if not self.embeddings_data:
            logger.warning("No embeddings data available")
            return
        
        try:
            # Extract embeddings and create mapping
            embeddings_list = []
            self.embedding_to_chunk_map = {}
            
            for i, emb_data in enumerate(self.embeddings_data):
                if 'embedding' in emb_data and emb_data['embedding']:
                    embeddings_list.append(emb_data['embedding'])
                    self.embedding_to_chunk_map[len(embeddings_list) - 1] = {
                        'chunk_id': emb_data.get('chunk_id', i),
                        'document_title': emb_data.get('document_title', 'Unknown')
                    }
            
            if embeddings_list:
                self.document_embeddings = np.array(embeddings_list)
                logger.info(f"Prepared embeddings matrix with shape: {self.document_embeddings.shape}")
            else:
                logger.warning("No valid embeddings found")
                
        except Exception as e:
            logger.error(f"Failed to prepare embeddings matrix: {e}")
            self.document_embeddings = None
    
    def search_documents(self, query: str, top_k: int = 5, min_similarity: float = 0.15) -> List[SearchResult]:
        """
        Search for relevant document chunks using semantic similarity.
        
        Args:
            query: The user's search query
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects sorted by similarity score
        """
        start_time = time.time()
        
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            # Generate query embedding using the same BGE model
            query_embedding = self.model.encode([query.strip()])
            
            # Use MongoDB vector search if available
            if self.use_mongodb and self.mongodb_collection_obj is not None:
                results = self._search_with_mongodb_vector_search(
                    query_embedding, top_k, min_similarity
                )
            # If we have pre-computed embeddings, use them
            elif self.document_embeddings is not None:
                results = self._search_with_precomputed_embeddings(
                    query_embedding, top_k, min_similarity
                )
            else:
                # Fallback: compute similarities on the fly
                results = self._search_with_realtime_embeddings(
                    query, query_embedding, top_k, min_similarity
                )
            
            processing_time = time.time() - start_time
            logger.info(f"Search completed in {processing_time:.3f}s, found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _search_with_precomputed_embeddings(self, query_embedding: np.ndarray, 
                                          top_k: int, min_similarity: float) -> List[SearchResult]:
        """Search using pre-computed document embeddings."""
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity < min_similarity:
                break
            
            # Get chunk data
            chunk_info = self.embedding_to_chunk_map.get(idx, {})
            chunk_id = chunk_info.get('chunk_id', idx)
            document_title = chunk_info.get('document_title', 'Unknown')
            
            # Find the corresponding chunk in chunks_data
            chunk_data = self._find_chunk_by_id(chunk_id, document_title)
            
            if chunk_data:
                result = SearchResult(
                    content=chunk_data['content'],
                    metadata=chunk_data['metadata'],
                    similarity_score=float(similarity),
                    source_document=document_title,
                    chunk_id=chunk_id
                )
                results.append(result)
        
        return results
    
    def _search_with_realtime_embeddings(self, query: str, query_embedding: np.ndarray,
                                       top_k: int, min_similarity: float) -> List[SearchResult]:
        """Search by computing embeddings in real-time."""
        if not self.chunks_data:
            return []
        
        similarities = []
        chunk_texts = []
        
        # Compute similarities for each chunk
        for chunk_data in self.chunks_data:
            content = chunk_data.get('content', '')
            if not content.strip():
                continue
            
            try:
                # Generate embedding for this chunk
                chunk_embedding = self.model.encode([content])
                similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
                
                if similarity >= min_similarity:
                    similarities.append(similarity)
                    chunk_texts.append(chunk_data)
            except Exception as e:
                logger.warning(f"Failed to compute similarity for chunk: {e}")
                continue
        
        # Sort by similarity and get top-k
        if not similarities:
            return []
        
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in sorted_indices:
            chunk_data = chunk_texts[idx]
            similarity = similarities[idx]
            
            result = SearchResult(
                content=chunk_data['content'],
                metadata=chunk_data['metadata'],
                similarity_score=float(similarity),
                source_document=chunk_data['metadata'].get('title', 'Unknown'),
                chunk_id=chunk_data['metadata'].get('chunk_id', 0)
            )
            results.append(result)
        
        return results
    
    def _search_with_mongodb_vector_search(self, query_embedding: np.ndarray, 
                                         top_k: int, min_similarity: float) -> List[SearchResult]:
        """
        Search for relevant documents using MongoDB with cosine similarity.
        
        Since Atlas Search is not configured, we'll fetch all documents and
        compute cosine similarity locally.
        
        Args:
            query_embedding: The query embedding vector
            top_k: Number of top results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SearchResult objects sorted by similarity score
        """
        try:
            # Convert numpy array to list for MongoDB
            query_vector = query_embedding[0].tolist()
            
            # Fetch all documents with embeddings from MongoDB
            cursor = self.mongodb_collection_obj.find(
                {"embedding": {"$exists": True, "$ne": None}},
                {
                    "content": 1,
                    "metadata": 1,
                    "document_title": 1,
                    "chunk_id": 1,
                    "embedding": 1
                }
            )
            
            # Convert to list for processing
            documents = list(cursor)
            logger.info(f"Retrieved {len(documents)} documents from MongoDB")
            
            if not documents:
                logger.warning("No documents with embeddings found in MongoDB")
                return []
            
            # Compute similarities
            similarities = []
            doc_embeddings = []
            
            for doc in documents:
                embedding = doc.get('embedding')
                if embedding and len(embedding) > 0:
                    doc_embeddings.append(embedding)
                    similarities.append(doc)
            
            if not doc_embeddings:
                logger.warning("No valid embeddings found in documents")
                return []
            
            # Convert to numpy arrays for similarity computation
            doc_embeddings_array = np.array(doc_embeddings)
            query_embedding_array = np.array(query_vector).reshape(1, -1)
            
            # Compute cosine similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities_scores = cosine_similarity(query_embedding_array, doc_embeddings_array)[0]
            
            # Create results with similarity scores
            results = []
            for i, (doc, similarity_score) in enumerate(zip(similarities, similarities_scores)):
                if similarity_score >= min_similarity:
                    # Extract document data
                    content = doc.get('content', '')
                    metadata = doc.get('metadata', {})
                    document_title = doc.get('document_title', 'Unknown')
                    chunk_id = doc.get('chunk_id', 0)
                    
                    # Create SearchResult object
                    result = SearchResult(
                        content=content,
                        metadata=metadata,
                        similarity_score=float(similarity_score),
                        source_document=document_title,
                        chunk_id=chunk_id
                    )
                    results.append(result)
            
            # Sort by similarity score (highest first) and limit to top_k
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            results = results[:top_k]
            
            logger.info(f"MongoDB similarity search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"MongoDB similarity search failed: {e}")
            logger.warning("Falling back to local search methods")
            
            # Fallback to local search if MongoDB fails
            if self.document_embeddings is not None:
                return self._search_with_precomputed_embeddings(
                    query_embedding, top_k, min_similarity
                )
            else:
                # This fallback requires the original query string
                # We'll need to pass it through or reconstruct it
                return []
    
    def _find_chunk_by_id(self, chunk_id: int, document_title: str) -> Optional[Dict[str, Any]]:
        """Find a specific chunk by its ID and document title."""
        for chunk_data in self.chunks_data:
            metadata = chunk_data.get('metadata', {})
            if (metadata.get('chunk_id') == chunk_id and 
                metadata.get('title') == document_title):
                return chunk_data
        return None
    
    def generate_response(self, query: str, top_k: int = 5, 
                         min_similarity: float = 0.15) -> RAGResponse:
        """
        Generate a comprehensive response to a user query.
        
        Args:
            query: The user's question or query
            top_k: Number of relevant chunks to retrieve
            min_similarity: Minimum similarity threshold for results
            
        Returns:
            RAGResponse object with answer, sources, and metadata
        """
        start_time = time.time()
        
        # Search for relevant documents
        search_results = self.search_documents(query, top_k, min_similarity)
        
        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question. Please try rephrasing your query or ask about topics related to AI bootcamp, training, or internships.",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Generate answer by combining relevant chunks
        answer = self._generate_answer_from_chunks(query, search_results)
        
        # Calculate confidence based on similarity scores
        confidence = self._calculate_confidence(search_results)
        
        processing_time = time.time() - start_time
        
        return RAGResponse(
            answer=answer,
            sources=search_results,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _generate_answer_from_chunks(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate a coherent answer from the retrieved chunks."""
        if not search_results:
            return "No relevant information found."
        
        # Sort results by similarity score
        sorted_results = sorted(search_results, key=lambda x: x.similarity_score, reverse=True)
        
        # Combine the most relevant chunks
        answer_parts = []
        answer_parts.append(f"Based on the available information, here's what I found about your question:\n")
        
        for i, result in enumerate(sorted_results[:3]):  # Use top 3 most relevant chunks
            content = result.content.strip()
            if len(content) > 500:  # Truncate very long chunks
                content = content[:500] + "..."
            
            answer_parts.append(f"**From {result.source_document}:**\n{content}\n")
        
        # Add source information
        if len(search_results) > 1:
            answer_parts.append(f"\n*Found {len(search_results)} relevant sections across the documents.*")
        
        return "\n".join(answer_parts)
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """Calculate confidence score based on similarity scores."""
        if not search_results:
            return 0.0
        
        # Use the highest similarity score as base confidence
        max_similarity = max(result.similarity_score for result in search_results)
        
        # Boost confidence if we have multiple relevant results
        result_count_bonus = min(len(search_results) * 0.1, 0.3)
        
        confidence = min(max_similarity + result_count_bonus, 1.0)
        return confidence
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get a summary of available documents and chunks."""
        if not self.chunks_data:
            return {"total_chunks": 0, "documents": []}
        
        # Count chunks per document
        doc_counts = {}
        for chunk_data in self.chunks_data:
            title = chunk_data.get('metadata', {}).get('title', 'Unknown')
            doc_counts[title] = doc_counts.get(title, 0) + 1
        
        return {
            "total_chunks": len(self.chunks_data),
            "total_documents": len(doc_counts),
            "documents": [
                {"title": title, "chunk_count": count}
                for title, count in doc_counts.items()
            ]
        }
    
    def reload_data(self) -> None:
        """Reload chunks and embeddings data from files."""
        logger.info("Reloading data...")
        self._load_data()
        logger.info("Data reloaded successfully")


def create_rag_service(use_mongodb: bool = False, 
                      mongodb_connection_string: Optional[str] = None) -> RAGService:
    """
    Factory function to create a RAG service instance.
    
    Args:
        use_mongodb: Whether to use MongoDB for vector search
        mongodb_connection_string: MongoDB connection string (required if use_mongodb=True)
    
    Returns:
        RAGService instance
    """
    if use_mongodb and mongodb_connection_string:
        return RAGService(
            use_mongodb=True,
            mongodb_connection_string=mongodb_connection_string
        )
    else:
        return RAGService()


def create_mongodb_rag_service(connection_string: str, 
                              database: str = "discord_rag", 
                              collection: str = "document_chunks") -> RAGService:
    """
    Factory function to create a MongoDB-enabled RAG service instance.
    
    Args:
        connection_string: MongoDB connection string
        database: MongoDB database name
        collection: MongoDB collection name
    
    Returns:
        RAGService instance configured for MongoDB vector search
    """
    return RAGService(
        use_mongodb=True,
        mongodb_connection_string=connection_string,
        mongodb_database=database,
        mongodb_collection=collection
    )


def main():
    """
    Main function that takes a user query as input and returns a response.
    This function can be called from other parts of the application.
    """
    import sys
    
    # Initialize the RAG service
    try:
        rag = create_rag_service()
        print("RAG Service initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize RAG service: {e}")
        return None
    
    # Get user query
    if len(sys.argv) > 1:
        # Query provided as command line argument
        query = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        query = input("Enter your question: ").strip()
    
    if not query:
        print("No query provided. Exiting.")
        return None
    
    # Generate response
    try:
        print(f"\nProcessing query: '{query}'")
        print("=" * 50)
        
        response = rag.generate_response(query)
        
        # Display results
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nConfidence: {response.confidence:.2f}")
        print(f"Sources found: {len(response.sources)}")
        print(f"Processing time: {response.processing_time:.3f}s")
        
        # Show sources if available
        if response.sources:
            print(f"\nSources:")
            for i, source in enumerate(response.sources[:3], 1):
                print(f"{i}. {source.source_document} (similarity: {source.similarity_score:.3f})")
        
        return response
        
    except Exception as e:
        print(f"Error processing query: {e}")
        return None


def process_user_query(query: str) -> Optional[RAGResponse]:
    """
    Process a user query and return a RAG response.
    
    Args:
        query: The user's question or query string
        
    Returns:
        RAGResponse object with answer and metadata, or None if error
    """
    try:
        # Initialize RAG service
        rag = create_rag_service()
        
        # Generate response
        response = rag.generate_response(query)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing user query '{query}': {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    # Test the RAG service
    rag = create_rag_service()
    
    # Test queries
    test_queries = [
        "What is the AI bootcamp about?",
        "How long is the training program?",
        "What are the requirements for AI engineers?",
        "Tell me about the internship program"
    ]
    
    print("RAG Service Test Results:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = rag.generate_response(query)
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Sources: {len(response.sources)}")
        print(f"Processing time: {response.processing_time:.3f}s")
        print("-" * 30)
    
    # Run main function for interactive/command line usage
    print("\n" + "=" * 50)
    print("Interactive Mode:")
    main()
