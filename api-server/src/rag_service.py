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
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from src.azure_ai_client import AzureAIFoundryClient, create_azure_ai_client

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
                 use_mongodb: bool = False,
                 use_azure_ai: bool = True,
                 azure_endpoint: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 azure_model_name: Optional[str] = None):
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
            use_azure_ai: Whether to use Azure AI Foundry for response generation
            azure_endpoint: Azure AI Foundry endpoint URL
            azure_api_key: Azure AI Foundry API key
            azure_model_name: Azure AI Foundry model name
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
        
        # Azure AI configuration
        self.use_azure_ai = use_azure_ai
        self.azure_endpoint = azure_endpoint
        self.azure_api_key = azure_api_key
        self.azure_model_name = azure_model_name
        self.azure_client = None
        
        # Initialize the service
        self._load_model()
        if self.use_azure_ai:
            self._initialize_azure_ai()
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
    
    def _initialize_azure_ai(self) -> None:
        """Initialize Azure AI Foundry client."""
        try:
            logger.info("Initializing Azure AI Foundry client...")
            self.azure_client = AzureAIFoundryClient(
                endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                model_name=self.azure_model_name
            )
            logger.info("Successfully initialized Azure AI Foundry client")
        except Exception as e:
            logger.error(f"Failed to initialize Azure AI client: {e}")
            logger.warning("Falling back to simple text-based response generation")
            self.use_azure_ai = False
    
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
            skipped_embeddings = 0
            
            logger.info(f"Processing {len(self.embeddings_data)} embedding entries")
            
            for i, emb_data in enumerate(self.embeddings_data):
                if 'embedding' in emb_data and emb_data['embedding']:
                    embedding_index = len(embeddings_list)  # Use current length
                    embeddings_list.append(emb_data['embedding'])
                    self.embedding_to_chunk_map[embedding_index] = {  # Use embedding_index
                        'chunk_id': emb_data.get('chunk_id', i),
                        'document_title': emb_data.get('document_title', 'Unknown')
                    }
                else:
                    skipped_embeddings += 1
                    logger.debug(f"Skipped embedding {i}: missing or empty embedding data")
            
            if embeddings_list:
                self.document_embeddings = np.array(embeddings_list)
                logger.info(f"Prepared embeddings matrix with shape: {self.document_embeddings.shape}")
                logger.info(f"Successfully processed {len(embeddings_list)} embeddings, skipped {skipped_embeddings}")
                
                # Validate embedding dimensions
                if len(embeddings_list) > 0:
                    embedding_dim = len(embeddings_list[0])
                    logger.debug(f"Embedding dimension: {embedding_dim}")
                    
                    # Check for consistent dimensions
                    inconsistent_dims = [i for i, emb in enumerate(embeddings_list) if len(emb) != embedding_dim]
                    if inconsistent_dims:
                        logger.warning(f"Found {len(inconsistent_dims)} embeddings with inconsistent dimensions")
            else:
                logger.warning("No valid embeddings found")
                
        except Exception as e:
            logger.error(f"Failed to prepare embeddings matrix: {e}")
            logger.error(f"Error type: {type(e).__name__}")
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
            logger.debug(f"Generating embedding for query: '{query.strip()}'")
            query_embedding = self.model.encode([query.strip()])
            logger.debug(f"Query embedding shape: {query_embedding.shape}")
            
            # Determine search strategy and log it
            search_strategy = "unknown"
            if self.use_mongodb and self.mongodb_collection_obj is not None:
                search_strategy = "MongoDB vector search"
                logger.debug(f"Using {search_strategy}")
                results = self._search_with_mongodb_vector_search(
                    query_embedding, top_k, min_similarity
                )
            elif self.document_embeddings is not None:
                search_strategy = "pre-computed embeddings"
                logger.debug(f"Using {search_strategy} (matrix shape: {self.document_embeddings.shape})")
                results = self._search_with_precomputed_embeddings(
                    query_embedding, top_k, min_similarity
                )
            else:
                search_strategy = "real-time embeddings"
                logger.debug(f"Using {search_strategy} (chunks: {len(self.chunks_data)})")
                results = self._search_with_realtime_embeddings(
                    query, query_embedding, top_k, min_similarity
                )
            
            processing_time = time.time() - start_time
            logger.info(f"Search completed in {processing_time:.3f}s using {search_strategy}, found {len(results)} results")
            
            # Log result quality metrics
            if results:
                similarity_scores = [r.similarity_score for r in results]
                logger.debug(f"Similarity scores: min={min(similarity_scores):.3f}, max={max(similarity_scores):.3f}, avg={sum(similarity_scores)/len(similarity_scores):.3f}")
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Search failed after {processing_time:.3f}s: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Query: '{query}', top_k: {top_k}, min_similarity: {min_similarity}")
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
            else:
                # Log when we can't find a chunk that should exist
                logger.warning(f"Embedding found for chunk_id={chunk_id}, document='{document_title}' but chunk data not found in chunks_data")
                logger.debug(f"This indicates a potential data inconsistency between embeddings and chunks")
        
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
            doc_embeddings = []
            valid_docs = []
            
            for doc in documents:
                embedding = doc.get('embedding')
                if embedding and len(embedding) > 0:
                    doc_embeddings.append(embedding)
                    valid_docs.append(doc)  # Store documents separately
            
            if not doc_embeddings:
                logger.warning("No valid embeddings found in documents")
                return []
            
            # Convert to numpy arrays for similarity computation
            doc_embeddings_array = np.array(doc_embeddings)
            query_embedding_array = np.array(query_vector).reshape(1, -1)
            
            # Compute cosine similarities
            similarities_scores = cosine_similarity(query_embedding_array, doc_embeddings_array)[0]
            
            # Create results with similarity scores
            results = []
            for i, (doc, similarity_score) in enumerate(zip(valid_docs, similarities_scores)):
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
        logger.debug(f"Searching for chunk_id={chunk_id}, document_title='{document_title}'")
        
        # Track search statistics for debugging
        total_chunks = len(self.chunks_data)
        matching_document_chunks = 0
        matching_id_chunks = 0
        
        for chunk_data in self.chunks_data:
            metadata = chunk_data.get('metadata', {})
            
            # Count chunks for debugging
            if metadata.get('title') == document_title:
                matching_document_chunks += 1
            if metadata.get('chunk_id') == chunk_id:
                matching_id_chunks += 1
            
            # Check for exact match
            if (metadata.get('chunk_id') == chunk_id and 
                metadata.get('title') == document_title):
                logger.debug(f"Found chunk: chunk_id={chunk_id}, document='{document_title}', content_length={len(chunk_data.get('content', ''))}")
                return chunk_data
        
        # Enhanced error logging with diagnostic information
        logger.warning(f"Chunk not found: chunk_id={chunk_id}, document_title='{document_title}'")
        logger.warning(f"Search diagnostics: total_chunks={total_chunks}, chunks_with_matching_document={matching_document_chunks}, chunks_with_matching_id={matching_id_chunks}")
        
        # Log available chunk IDs and document titles for debugging
        if total_chunks > 0:
            available_chunk_ids = [chunk.get('metadata', {}).get('chunk_id') for chunk in self.chunks_data[:5]]
            available_titles = list(set([chunk.get('metadata', {}).get('title') for chunk in self.chunks_data]))
            logger.debug(f"Available chunk_ids (first 5): {available_chunk_ids}")
            logger.debug(f"Available document titles: {available_titles}")
        
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
        
        # Check if this is a simple greeting
        if self._is_greeting(query):
            return self._generate_greeting_response(query, start_time)
        
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
    
    def _is_greeting(self, query: str) -> bool:
        """
        Check if the query is a simple greeting.
        
        Args:
            query: The user's query
            
        Returns:
            True if it's a greeting, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Common greetings
        greetings = [
            'hi', 'hey', 'hello', 'hiya', 'howdy', 'greetings',
            'good morning', 'good afternoon', 'good evening',
            'what\'s up', 'whats up', 'sup', 'yo',
            'good day', 'how are you', 'how do you do'
        ]
        
        # Check for exact matches or very short queries
        if query_lower in greetings:
            return True
            
        # Check for queries that are just greetings (very short and simple)
        if len(query_lower.split()) <= 2 and any(greeting in query_lower for greeting in greetings):
            return True
            
        return False
    
    def _generate_greeting_response(self, query: str, start_time: float) -> RAGResponse:
        """
        Generate a friendly greeting response.
        
        Args:
            query: The user's greeting
            start_time: Start time for processing time calculation
            
        Returns:
            RAGResponse with greeting message
        """
        greeting_responses = [
            "Hello! 👋 I'm your AI Bootcamp assistant. I can help you with questions about:",
            "Hi there! 🤖 Welcome! I'm here to help you with AI Bootcamp information. I can assist with:",
            "Hey! 😊 Great to meet you! I'm your AI Bootcamp FAQ bot. I can help you learn about:",
            "Hello! 🚀 Nice to see you! I'm here to answer questions about the AI Bootcamp program. I can help with:"
        ]
        
        import random
        greeting = random.choice(greeting_responses)
        
        help_topics = [
            "• **Program Details**: Duration, structure, and timeline",
            "• **Team Matching**: How to join teams and collaborate", 
            "• **Technologies**: What you'll learn and work with",
            "• **Requirements**: Deadlines, sessions, and expectations",
            "• **Communication**: Discord channels and protocols",
            "• **Visa Support**: Sponsorship and documentation help"
        ]
        
        response_text = f"{greeting}\n\n" + "\n".join(help_topics) + "\n\n**What would you like to know about?** Just ask me anything! 😊"
        
        processing_time = time.time() - start_time
        
        return RAGResponse(
            answer=response_text,
            sources=[],
            confidence=1.0,  # High confidence for greetings
            processing_time=processing_time
        )
    
    def _generate_answer_from_chunks(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate a coherent answer from the retrieved chunks using LLM integration."""
        if not search_results:
            return "No relevant information found."
        
        # Sort results by similarity score
        sorted_results = sorted(search_results, key=lambda x: x.similarity_score, reverse=True)
        
        # Try OpenAI API first (more commonly available)
        openai_answer = self._generate_openai_response(query, sorted_results)
        if openai_answer:
            return openai_answer
        
        # Use Azure AI Foundry DeepSeek if available
        if self.use_azure_ai and self.azure_client:
            try:
                # Enhanced context preparation for better LLM understanding
                context_chunks = self._prepare_enhanced_context_chunks(sorted_results)
                
                logger.info(f"Generating response with {len(context_chunks)} context chunks")
                logger.debug(f"Query: {query}")
                logger.debug(f"Top similarity scores: {[r.similarity_score for r in sorted_results[:3]]}")
                
                # Generate response using enhanced DeepSeek integration
                response = self.azure_client.generate_rag_response(
                    query=query,
                    context_chunks=context_chunks,
                    max_tokens=1200,  # Increased for more comprehensive responses
                    temperature=0.7,
                    include_sources=True
                )
                
                # Process and enhance the response
                enhanced_answer = self._enhance_llm_response(response.content, sorted_results)
                
                logger.info("Successfully generated response using DeepSeek")
                return enhanced_answer
                
            except Exception as e:
                logger.error(f"DeepSeek generation failed: {e}")
                logger.warning("Falling back to simple text-based response generation")
                # Fall through to simple text generation
        
        # Fallback: Simple text-based response generation
        return self._generate_fallback_response(query, sorted_results)
    
    def _prepare_enhanced_context_chunks(self, search_results: List[SearchResult]) -> List[str]:
        """
        Prepare context chunks with enhanced formatting for better LLM understanding.
        
        Args:
            search_results: List of SearchResult objects sorted by relevance
            
        Returns:
            List of formatted context chunk strings
        """
        context_chunks = []
        
        for i, result in enumerate(search_results[:5]):  # Use top 5 most relevant chunks
            content = result.content.strip()
            
            # Truncate very long chunks to prevent token overflow
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            # Create enhanced context chunk with metadata
            enhanced_chunk = f"""Source Document: {result.source_document}
Relevance Score: {result.similarity_score:.3f}
Content: {content}"""
            
            context_chunks.append(enhanced_chunk)
        
        return context_chunks
    
    def _enhance_llm_response(self, llm_response: str, search_results: List[SearchResult]) -> str:
        """
        Enhance the LLM response with additional metadata and source information.
        
        Args:
            llm_response: The response generated by the LLM
            search_results: List of SearchResult objects used for context
            
        Returns:
            Enhanced response string
        """
        # Add source information
        source_info = f"\n\n---\n*Response generated using {len(search_results)} relevant document sections*"
        
        # Add confidence indicator based on similarity scores
        if search_results:
            avg_confidence = sum(r.similarity_score for r in search_results[:3]) / min(3, len(search_results))
            confidence_level = "High" if avg_confidence > 0.7 else "Medium" if avg_confidence > 0.4 else "Low"
            confidence_info = f"\n*Confidence Level: {confidence_level} (avg similarity: {avg_confidence:.2f})*"
            source_info += confidence_info
        
        return llm_response + source_info
    
    def _generate_fallback_response(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Generate a fallback response when LLM is not available.
        
        Args:
            query: The user's question
            search_results: List of SearchResult objects
            
        Returns:
            Fallback response string
        """
        answer_parts = []
        answer_parts.append(f"Based on the available information, here's what I found about your question:\n")
        
        for i, result in enumerate(search_results[:3]):  # Use top 3 most relevant chunks
            content = result.content.strip()
            if len(content) > 500:  # Truncate very long chunks
                content = content[:500] + "..."
            
            answer_parts.append(f"**From {result.source_document} (relevance: {result.similarity_score:.2f}):**\n{content}\n")
        
        # Add source information
        if len(search_results) > 1:
            answer_parts.append(f"\n*Found {len(search_results)} relevant sections across the documents.*")
        
        return "\n".join(answer_parts)
    
    def _generate_openai_response(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Generate a response using OpenAI API.
        
        Args:
            query: The user's question
            search_results: List of relevant search results
            
        Returns:
            Generated response string or None if OpenAI is not available
        """
        try:
            import openai
            
            # Get OpenAI API key from environment
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key or openai_api_key == "your-openai-api-key":
                logger.debug("OpenAI API key not configured")
                return None
            
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=openai_api_key)
            
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results[:3], 1):  # Use top 3 results
                content = result.content.strip()
                if len(content) > 800:  # Truncate very long chunks
                    content = content[:800] + "..."
                context_parts.append(f"Source {i} (from {result.source_document}):\n{content}")
            
            context = "\n\n".join(context_parts)
            
            # Create the prompt
            system_prompt = """You are a helpful AI assistant that answers questions using provided context documents.

Instructions:
- Base your answer ONLY on the provided context
- Be accurate and specific
- If the context doesn't contain enough information, say so clearly
- Synthesize information from multiple sources when relevant
- Be conversational but professional
- Don't copy text verbatim - explain and summarize in your own words"""

            user_prompt = f"""Context:
{context}

Question: {query}

Please answer the question using the context above. If the context doesn't contain enough information, let me know."""

            # Make API call
            logger.info("Generating response using OpenAI API")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Successfully generated response using OpenAI")
            return answer
            
        except ImportError:
            logger.debug("OpenAI library not installed")
            return None
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return None
    
    def generate_enhanced_response(self, query: str, top_k: int = 5, 
                                 min_similarity: float = 0.15,
                                 max_tokens: int = 1200,
                                 temperature: float = 0.7) -> RAGResponse:
        """
        Generate an enhanced response using the improved LLM integration.
        
        This method demonstrates the enhanced RAG pipeline with:
        - Better prompt construction
        - Structured context formatting
        - Enhanced API call handling
        - Improved response processing
        
        Args:
            query: The user's question or query
            top_k: Number of relevant chunks to retrieve
            min_similarity: Minimum similarity threshold for results
            max_tokens: Maximum tokens for LLM response
            temperature: Sampling temperature for LLM
            
        Returns:
            RAGResponse object with enhanced answer, sources, and metadata
        """
        start_time = time.time()
        
        logger.info(f"Processing enhanced RAG query: '{query}'")
        
        # Search for relevant documents
        search_results = self.search_documents(query, top_k, min_similarity)
        
        if not search_results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question. Please try rephrasing your query or ask about topics related to AI bootcamp, training, or internships.",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time
            )
        
        # Generate enhanced answer using improved LLM integration
        answer = self._generate_enhanced_answer(query, search_results, max_tokens, temperature)
        
        # Calculate confidence based on similarity scores
        confidence = self._calculate_confidence(search_results)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Enhanced RAG response generated in {processing_time:.3f}s with confidence {confidence:.2f}")
        
        return RAGResponse(
            answer=answer,
            sources=search_results,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _generate_enhanced_answer(self, query: str, search_results: List[SearchResult], 
                                max_tokens: int, temperature: float) -> str:
        """
        Generate an enhanced answer using the improved LLM integration.
        
        This method showcases the enhanced RAG pipeline:
        1. Constructs structured prompts with context
        2. Makes optimized API calls to the LLM
        3. Processes and enhances the generated response
        
        Args:
            query: The user's question
            search_results: List of relevant search results
            max_tokens: Maximum tokens for LLM response
            temperature: Sampling temperature
            
        Returns:
            Enhanced answer string
        """
        # Sort results by similarity score
        sorted_results = sorted(search_results, key=lambda x: x.similarity_score, reverse=True)
        
        # Use enhanced Azure AI Foundry DeepSeek if available
        if self.use_azure_ai and self.azure_client:
            try:
                logger.info("Using enhanced DeepSeek integration for response generation")
                
                # Prepare enhanced context chunks
                context_chunks = self._prepare_enhanced_context_chunks(sorted_results)
                
                # Make the enhanced API call to the LLM
                logger.info(f"Making enhanced API call with {len(context_chunks)} context chunks")
                response = self.azure_client.generate_rag_response(
                    query=query,
                    context_chunks=context_chunks,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    include_sources=True
                )
                
                # Process and enhance the LLM response
                enhanced_answer = self._enhance_llm_response(response.content, sorted_results)
                
                logger.info("Successfully generated enhanced response using DeepSeek")
                return enhanced_answer
                
            except Exception as e:
                logger.error(f"Enhanced DeepSeek generation failed: {e}")
                logger.warning("Falling back to enhanced text-based response generation")
                # Fall through to enhanced fallback
        
        # Enhanced fallback response
        return self._generate_fallback_response(query, sorted_results)
    
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
                      mongodb_connection_string: Optional[str] = None,
                      use_azure_ai: bool = True,
                      azure_endpoint: Optional[str] = None,
                      azure_api_key: Optional[str] = None,
                      azure_model_name: Optional[str] = None) -> RAGService:
    """
    Factory function to create a RAG service instance.
    
    Args:
        use_mongodb: Whether to use MongoDB for vector search
        mongodb_connection_string: MongoDB connection string (required if use_mongodb=True)
        use_azure_ai: Whether to use Azure AI Foundry for response generation
        azure_endpoint: Azure AI Foundry endpoint URL
        azure_api_key: Azure AI Foundry API key
        azure_model_name: Azure AI Foundry model name
    
    Returns:
        RAGService instance
    """
    return RAGService(
        use_mongodb=use_mongodb,
        mongodb_connection_string=mongodb_connection_string,
        use_azure_ai=use_azure_ai,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_model_name=azure_model_name
    )


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


