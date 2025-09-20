#!/usr/bin/env python3
"""
Document ingestion and chunking pipeline for Discord RAG Chatbot.

This module provides a robust, scalable solution for downloading Google Docs
and implementing intelligent document chunking using LangChain. It features
comprehensive error handling, security best practices, and industry-standard
code organization.

Author: AI Assistant
Version: 2.0.0
License: MIT
"""

import os
import json
import logging
import hashlib
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import urlparse, parse_qs
import time
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# Embedding imports
from sentence_transformers import SentenceTransformer
import numpy as np

# LangChain imports for document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader

# Google Docs API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class ProcessingStatus(Enum):
    """Status enumeration for document processing."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    CHUNKING = "chunking"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class DocumentSource(Enum):
    """Source types for documents."""
    GOOGLE_DOCS = "google_docs"
    WEB_SCRAPING = "web_scraping"
    API = "api"

@dataclass
class DocumentMetadata:
    """Structured metadata for documents."""
    source: str
    title: str
    document_id: int
    content_length: int
    chunk_count: int
    processing_status: ProcessingStatus
    download_method: DocumentSource
    created_at: str
    updated_at: str
    checksum: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enums to their values for JSON serialization
        result['processing_status'] = self.processing_status.value
        result['download_method'] = self.download_method.value
        return result

@dataclass
class ChunkMetadata:
    """Structured metadata for document chunks."""
    chunk_id: int
    total_chunks: int
    chunk_size: int
    source_document: str
    document_title: str
    chunk_hash: str
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retries: int = 3
    timeout_seconds: int = 30
    output_dir: str = "source_documents"
    enable_checksums: bool = True
    enable_compression: bool = False
    log_level: str = "INFO"

# =============================================================================
# EXCEPTIONS
# =============================================================================

class DocumentIngestionError(Exception):
    """Base exception for document ingestion errors."""
    pass

class DownloadError(DocumentIngestionError):
    """Exception raised when document download fails."""
    pass

class ChunkingError(DocumentIngestionError):
    """Exception raised when document chunking fails."""
    pass

class ConfigurationError(DocumentIngestionError):
    """Exception raised when configuration is invalid."""
    pass

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_checksum(content: str) -> str:
    """Generate SHA-256 checksum for content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem usage."""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized

def validate_url(url: str) -> bool:
    """Validate if URL is a proper Google Docs URL."""
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme in ['http', 'https'] and
            'docs.google.com' in parsed.netloc and
            '/document/d/' in parsed.path
        )
    except Exception:
        return False

def create_secure_session() -> requests.Session:
    """Create a secure HTTP session with retry strategy."""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set secure headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (compatible; DocumentIngestion/2.0)',
        'Accept': 'text/plain,text/html,application/json',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    })
    
    return session

# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================

class DocumentDownloader(ABC):
    """Abstract base class for document downloaders."""
    
    @abstractmethod
    def download(self, url: str) -> Tuple[str, DocumentSource]:
        """Download document content from URL."""
        pass
    
    @abstractmethod
    def is_supported(self, url: str) -> bool:
        """Check if URL is supported by this downloader."""
        pass

class DocumentChunker(ABC):
    """Abstract base class for document chunkers."""
    
    @abstractmethod
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split document content into chunks."""
        pass

class EmbeddingService:
    """Service for generating embeddings using BGE-base-en-v1.5 model."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """Initialize the embedding service with the specified model."""
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the embedding model with error handling."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise DocumentIngestionError(f"Could not load embedding model: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text chunk."""
        if not self.model:
            raise DocumentIngestionError("Embedding model not loaded")
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            return []
        
        try:
            # Clean and prepare text
            cleaned_text = text.strip()
            if len(cleaned_text) == 0:
                return []
            
            # Generate embedding
            embedding = self.model.encode(cleaned_text, convert_to_tensor=False)
            
            # Convert numpy array to list for JSON serialization
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            logger.debug(f"Generated embedding of dimension {len(embedding)} for text of length {len(cleaned_text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise DocumentIngestionError(f"Embedding generation failed: {e}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple text chunks efficiently."""
        if not self.model:
            raise DocumentIngestionError("Embedding model not loaded")
        
        if not texts:
            return []
        
        try:
            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                logger.warning("No valid texts provided for batch embedding generation")
                return [[] for _ in texts]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(valid_texts, convert_to_tensor=False, batch_size=32)
            
            # Convert numpy arrays to lists
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Handle case where some texts were filtered out
            result = []
            valid_idx = 0
            for text in texts:
                if text and text.strip():
                    result.append(embeddings[valid_idx])
                    valid_idx += 1
                else:
                    result.append([])
            
            logger.info(f"Generated {len(result)} embeddings in batch")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise DocumentIngestionError(f"Batch embedding generation failed: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if not self.model:
            raise DocumentIngestionError("Embedding model not loaded")
        
        # Generate a test embedding to get dimension
        test_embedding = self.generate_embedding("test")
        return len(test_embedding)

# =============================================================================
# CONCRETE IMPLEMENTATIONS
# =============================================================================

class GoogleDocsDownloader(DocumentDownloader):
    """Secure, robust Google Docs downloader with API and fallback support."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.service = None
        self.credentials = None
        self.session = create_secure_session()
        self._setup_google_docs_api()
    
    def _setup_google_docs_api(self) -> None:
        """Setup Google Docs API credentials and service with security best practices."""
        SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
        creds_file = os.getenv('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
        token_file = os.getenv('GOOGLE_TOKEN_FILE', 'token.json')
        
        # Validate credentials file exists and is readable
        if not os.path.exists(creds_file):
            logger.warning(f"Google credentials file '{creds_file}' not found. "
                         "Falling back to web scraping method.")
            return
        
        try:
            # Load existing credentials
            if os.path.exists(token_file):
                self.credentials = Credentials.from_authorized_user_file(token_file, SCOPES)
            
            # Refresh or create new credentials
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())
                else:
                    flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
                    self.credentials = flow.run_local_server(port=0)
                
                # Save credentials securely
                with open(token_file, 'w') as token:
                    token.write(self.credentials.to_json())
                os.chmod(token_file, 0o600)  # Restrict permissions
            
            # Initialize service
            self.service = build('docs', 'v1', credentials=self.credentials)
            logger.info("Google Docs API service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Docs API: {e}")
            self.service = None
    
    def extract_document_id(self, url: str) -> Optional[str]:
        """Extract document ID from Google Docs URL with validation."""
        if not validate_url(url):
            logger.error(f"Invalid Google Docs URL format: {url}")
            return None
        
        try:
            # Extract ID using regex for better reliability
            pattern = r'/document/d/([a-zA-Z0-9-_]+)'
            match = re.search(pattern, url)
            if match:
                doc_id = match.group(1)
                logger.debug(f"Extracted document ID: {doc_id}")
                return doc_id
            else:
                logger.error(f"Could not extract document ID from URL: {url}")
                return None
        except Exception as e:
            logger.error(f"Error extracting document ID from {url}: {e}")
            return None
    
    def download(self, url: str) -> Tuple[str, DocumentSource]:
        """Download document content with fallback strategy."""
        if not self.is_supported(url):
            raise DownloadError(f"Unsupported URL format: {url}")
        
        # Try API first
        if self.service:
            try:
                content = self._download_via_api(url)
                if content and not content.startswith("Error"):
                    return content, DocumentSource.API
            except Exception as e:
                logger.warning(f"API download failed, trying fallback: {e}")
        
        # Fallback to web scraping
        content = self._download_via_web_scraping(url)
        return content, DocumentSource.WEB_SCRAPING
    
    def _download_via_api(self, url: str) -> str:
        """Download document using Google Docs API."""
        doc_id = self.extract_document_id(url)
        if not doc_id:
            raise DownloadError(f"Could not extract document ID from {url}")
        
        try:
            document = self.service.documents().get(documentId=doc_id).execute()
            
            # Extract text content with better error handling
            content = []
            for element in document.get('body', {}).get('content', []):
                if 'paragraph' in element:
                    paragraph = element['paragraph']
                    for text_run in paragraph.get('elements', []):
                        if 'textRun' in text_run:
                            content.append(text_run['textRun']['content'])
            
            result = ''.join(content)
            if not result.strip():
                raise DownloadError("Document appears to be empty")
            
            logger.info(f"Successfully downloaded document via API: {len(result)} characters")
            return result
            
        except HttpError as e:
            if e.resp.status == 403:
                raise DownloadError(f"Access denied to document: {url}")
            elif e.resp.status == 404:
                raise DownloadError(f"Document not found: {url}")
            else:
                raise DownloadError(f"Google Docs API error: {e}")
        except Exception as e:
            raise DownloadError(f"Unexpected error downloading document: {e}")
    
    def _download_via_web_scraping(self, url: str) -> str:
        """Download document using web scraping with security measures."""
        doc_id = self.extract_document_id(url)
        if not doc_id:
            raise DownloadError(f"Could not extract document ID from {url}")
        
        try:
            export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
            
            response = self.session.get(
                export_url, 
                timeout=self.config.timeout_seconds,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Validate content
            content = response.text
            if not content.strip():
                raise DownloadError("Downloaded document appears to be empty")
            
            logger.info(f"Successfully downloaded document via web scraping: {len(content)} characters")
            return content
            
        except requests.exceptions.Timeout:
            raise DownloadError(f"Download timeout for document: {url}")
        except requests.exceptions.RequestException as e:
            raise DownloadError(f"Network error downloading document: {e}")
        except Exception as e:
            raise DownloadError(f"Unexpected error in web scraping: {e}")
    
    def is_supported(self, url: str) -> bool:
        """Check if URL is supported by this downloader."""
        return validate_url(url)

class LangChainDocumentChunker(DocumentChunker):
    """Advanced document chunker using LangChain with comprehensive metadata and embeddings."""
    
    def __init__(self, config: ProcessingConfig, embedding_service: Optional[EmbeddingService] = None):
        self.config = config
        self.embedding_service = embedding_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Added sentence boundaries
        )
    
    def chunk(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split document content into chunks with comprehensive metadata and embeddings."""
        if not content or not content.strip():
            raise ChunkingError("Cannot chunk empty content")
        
        try:
            # Validate input
            if len(content) < 10:
                logger.warning("Document content is very short, creating single chunk")
                return self._create_single_chunk(content, metadata)
            
            # Create base document
            doc = Document(page_content=content, metadata=metadata.copy())
            
            # Split the document
            chunks = self.text_splitter.split_documents([doc])
            
            if not chunks:
                raise ChunkingError("Chunking produced no results")
            
            # Generate embeddings if embedding service is available
            if self.embedding_service:
                logger.info(f"Generating embeddings for {len(chunks)} chunks")
                chunk_texts = [chunk.page_content for chunk in chunks]
                embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)
            else:
                logger.warning("No embedding service provided, skipping embedding generation")
                embeddings = [None] * len(chunks)
            
            # Enhance chunks with metadata and embeddings
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = self._enhance_chunk_metadata(chunk, i, len(chunks), embeddings[i])
                enhanced_chunks.append(enhanced_chunk)
            
            logger.info(f"Successfully chunked document into {len(enhanced_chunks)} chunks with embeddings")
            return enhanced_chunks
            
        except Exception as e:
            raise ChunkingError(f"Failed to chunk document: {e}")
    
    def _create_single_chunk(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Create a single chunk for very short content."""
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_id': 0,
            'total_chunks': 1,
            'chunk_size': len(content),
            'chunk_hash': generate_checksum(content) if self.config.enable_checksums else None
        })
        
        # Generate embedding for single chunk if embedding service is available
        if self.embedding_service:
            try:
                embedding = self.embedding_service.generate_embedding(content)
                chunk_metadata['embedding'] = embedding
            except Exception as e:
                logger.warning(f"Failed to generate embedding for single chunk: {e}")
                chunk_metadata['embedding'] = None
        else:
            chunk_metadata['embedding'] = None
        
        return [Document(page_content=content, metadata=chunk_metadata)]
    
    def _enhance_chunk_metadata(self, chunk: Document, chunk_id: int, total_chunks: int, embedding: Optional[List[float]] = None) -> Document:
        """Enhance chunk with comprehensive metadata and embedding."""
        enhanced_metadata = chunk.metadata.copy()
        enhanced_metadata.update({
            'chunk_id': chunk_id,
            'total_chunks': total_chunks,
            'chunk_size': len(chunk.page_content),
            'chunk_hash': generate_checksum(chunk.page_content) if self.config.enable_checksums else None,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'chunk_type': 'text',
            'language': 'en',  # Could be detected in future versions
            'embedding': embedding
        })
        
        return Document(page_content=chunk.page_content, metadata=enhanced_metadata)

class DocumentIngestionPipeline:
    """Enterprise-grade document ingestion pipeline with comprehensive error handling and embeddings."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None, enable_embeddings: bool = True):
        self.config = config or ProcessingConfig()
        self.enable_embeddings = enable_embeddings
        self._validate_config()
        
        # Setup output directory with proper permissions
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True, mode=0o755)
        
        # Initialize embedding service if enabled
        self.embedding_service = None
        if self.enable_embeddings:
            try:
                self.embedding_service = EmbeddingService()
                logger.info("Embedding service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding service: {e}")
                logger.warning("Continuing without embeddings...")
                self.embedding_service = None
        
        # Initialize components
        self.downloader = GoogleDocsDownloader(self.config)
        self.chunker = LangChainDocumentChunker(self.config, self.embedding_service)
        
        # Document configuration
        self.document_urls = [
            "https://docs.google.com/document/d/18O8Xpbognhi3GYJeHDYbBRHGAl5-a4DL/edit?usp=sharing&ouid=108974253833672780221&rtpof=true&sd=true",
            "https://docs.google.com/document/d/1NfmEDyxrJ7Tz7Wq4lAJHx1fQ4bPpM5v07dPTM4pjOsM/edit?usp=sharing",
            "https://docs.google.com/document/d/1RDsjDgszRw5yvjca24aqhmQu2UCrdWdVB3Bb_jd6V6Q/edit?usp=sharing"
        ]
        
        self.document_titles = [
            "AI_Bootcamp_Journey_Learning_Path",
            "Training_For_AI_Engineer_Interns", 
            "Intern_FAQ_AI_Bootcamp"
        ]
        
        # Processing statistics
        self.stats = {
            'total_documents': len(self.document_urls),
            'processed_documents': 0,
            'failed_documents': 0,
            'total_chunks': 0,
            'processing_time': 0
        }
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.chunk_size < 50:
            raise ConfigurationError("Chunk size must be at least 50 characters")
        if self.config.chunk_overlap < 0:
            raise ConfigurationError("Chunk overlap cannot be negative")
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ConfigurationError("Chunk overlap must be less than chunk size")
        if self.config.timeout_seconds < 5:
            raise ConfigurationError("Timeout must be at least 5 seconds")
    
    def process_documents(self) -> Dict[str, Any]:
        """Process all documents with comprehensive error handling and statistics."""
        start_time = time.time()
        logger.info("Starting enterprise document ingestion pipeline...")
        
        all_chunks = []
        processing_results = []
        
        for i, (url, title) in enumerate(zip(self.document_urls, self.document_titles)):
            try:
                logger.info(f"Processing document {i+1}/{len(self.document_urls)}: {title}")
                result = self._process_single_document(url, title, i)
                
                if result['status'] == ProcessingStatus.COMPLETED:
                    all_chunks.extend(result['chunks'])
                    self.stats['processed_documents'] += 1
                    self.stats['total_chunks'] += len(result['chunks'])
                else:
                    self.stats['failed_documents'] += 1
                
                processing_results.append(result)
                
            except Exception as e:
                logger.error(f"Unexpected error processing {title}: {e}")
                self.stats['failed_documents'] += 1
                processing_results.append({
                    'title': title,
                    'url': url,
                    'status': ProcessingStatus.FAILED,
                    'error': str(e),
                    'chunks': []
                })
        
        # Save all chunks and generate reports
        self._save_chunks(all_chunks)
        self._generate_processing_report(processing_results)
        
        # Update final statistics
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(f"Document ingestion complete! "
                   f"Processed {self.stats['processed_documents']}/{self.stats['total_documents']} documents, "
                   f"Generated {self.stats['total_chunks']} chunks in {self.stats['processing_time']:.2f}s")
        
        return {
            'stats': self.stats,
            'results': processing_results,
            'total_chunks': len(all_chunks)
        }
    
    def _process_single_document(self, url: str, title: str, doc_id: int) -> Dict[str, Any]:
        """Process a single document with comprehensive error handling."""
        try:
            # Download document
            content, download_method = self.downloader.download(url)
            
            # Generate document metadata
            metadata = self._create_document_metadata(
                url, title, doc_id, content, download_method
            )
            
            # Chunk the document
            chunks = self.chunker.chunk(content, metadata.to_dict())
            
            # Save individual document
            self._save_document(title, content, metadata)
            
            return {
                'title': title,
                'url': url,
                'status': ProcessingStatus.COMPLETED,
                'chunks': chunks,
                'metadata': metadata.to_dict(),
                'download_method': download_method.value
            }
            
        except DownloadError as e:
            logger.error(f"Download failed for {title}: {e}")
            return {
                'title': title,
                'url': url,
                'status': ProcessingStatus.FAILED,
                'error': str(e),
                'chunks': []
            }
        except ChunkingError as e:
            logger.error(f"Chunking failed for {title}: {e}")
            return {
                'title': title,
                'url': url,
                'status': ProcessingStatus.FAILED,
                'error': str(e),
                'chunks': []
            }
        except Exception as e:
            logger.error(f"Unexpected error processing {title}: {e}")
            return {
                'title': title,
                'url': url,
                'status': ProcessingStatus.FAILED,
                'error': str(e),
                'chunks': []
            }
    
    def _create_document_metadata(self, url: str, title: str, doc_id: int, 
                                content: str, download_method: DocumentSource) -> DocumentMetadata:
        """Create comprehensive document metadata."""
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        checksum = generate_checksum(content) if self.config.enable_checksums else None
        
        return DocumentMetadata(
            source=url,
            title=title,
            document_id=doc_id,
            content_length=len(content),
            chunk_count=0,  # Will be updated after chunking
            processing_status=ProcessingStatus.COMPLETED,
            download_method=download_method,
            created_at=current_time,
            updated_at=current_time,
            checksum=checksum
        )
    
    def _save_document(self, title: str, content: str, metadata: DocumentMetadata) -> None:
        """Save individual document content with security measures."""
        try:
            # Sanitize filename
            safe_title = sanitize_filename(title)
            
            # Save raw content
            content_file = self.output_dir / f"{safe_title}.txt"
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(content)
            os.chmod(content_file, 0o644)  # Set appropriate permissions
            
            # Save metadata
            metadata_file = self.output_dir / f"{safe_title}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            os.chmod(metadata_file, 0o644)
            
            logger.debug(f"Saved document: {safe_title}")
            
        except Exception as e:
            logger.error(f"Failed to save document {title}: {e}")
            raise
    
    def _save_chunks(self, chunks: List[Document]) -> None:
        """Save all document chunks with comprehensive metadata and embeddings."""
        try:
            chunks_data = []
            embeddings_data = []
            embedding_count = 0
            
            for chunk in chunks:
                chunks_data.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata
                })
                
                # Collect embeddings separately for easier access
                if chunk.metadata.get('embedding') is not None:
                    embeddings_data.append({
                        'chunk_id': chunk.metadata.get('chunk_id'),
                        'document_title': chunk.metadata.get('title'),
                        'embedding': chunk.metadata.get('embedding')
                    })
                    embedding_count += 1
            
            # Save chunks as JSON with backup
            chunks_file = self.output_dir / "all_chunks.json"
            backup_file = self.output_dir / "all_chunks.json.backup"
            
            # Create backup if file exists
            if chunks_file.exists():
                chunks_file.rename(backup_file)
            
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            os.chmod(chunks_file, 0o644)
            
            # Save embeddings separately if any exist
            if embeddings_data:
                embeddings_file = self.output_dir / "embeddings.json"
                with open(embeddings_file, 'w', encoding='utf-8') as f:
                    json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
                os.chmod(embeddings_file, 0o644)
                logger.info(f"Saved {embedding_count} embeddings to {embeddings_file}")
            
            # Save individual chunk files
            self._save_individual_chunks(chunks)
            
            # Save chunk index for quick access
            self._save_chunk_index(chunks)
            
            logger.info(f"Saved {len(chunks)} chunks to {chunks_file} ({embedding_count} with embeddings)")
            
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            raise
    
    def _save_individual_chunks(self, chunks: List[Document]) -> None:
        """Save individual chunk files for easy reading."""
        chunks_text_dir = self.output_dir / "chunks"
        chunks_text_dir.mkdir(exist_ok=True, mode=0o755)
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_file = chunks_text_dir / f"chunk_{i:04d}.txt"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    f.write(f"Source: {chunk.metadata.get('title', 'Unknown')}\n")
                    f.write(f"Chunk {chunk.metadata.get('chunk_id', i)} of {chunk.metadata.get('total_chunks', 'Unknown')}\n")
                    f.write(f"Size: {chunk.metadata.get('chunk_size', 0)} characters\n")
                    f.write(f"Hash: {chunk.metadata.get('chunk_hash', 'N/A')}\n")
                    f.write("-" * 50 + "\n")
                    f.write(chunk.page_content)
                os.chmod(chunk_file, 0o644)
            except Exception as e:
                logger.warning(f"Failed to save individual chunk {i}: {e}")
    
    def _save_chunk_index(self, chunks: List[Document]) -> None:
        """Save chunk index for quick access and search."""
        index_data = {
            'total_chunks': len(chunks),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'chunks': []
        }
        
        for i, chunk in enumerate(chunks):
            index_data['chunks'].append({
                'chunk_id': i,
                'title': chunk.metadata.get('title', 'Unknown'),
                'size': chunk.metadata.get('chunk_size', 0),
                'hash': chunk.metadata.get('chunk_hash', 'N/A'),
                'file': f"chunk_{i:04d}.txt"
            })
        
        index_file = self.output_dir / "chunk_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        os.chmod(index_file, 0o644)
    
    def _generate_processing_report(self, results: List[Dict[str, Any]]) -> None:
        """Generate comprehensive processing report."""
        # Convert results to JSON-serializable format
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            if 'status' in serializable_result and isinstance(serializable_result['status'], ProcessingStatus):
                serializable_result['status'] = serializable_result['status'].value
            # Convert Document objects to dictionaries
            if 'chunks' in serializable_result:
                serializable_chunks = []
                for chunk in serializable_result['chunks']:
                    if hasattr(chunk, 'page_content') and hasattr(chunk, 'metadata'):
                        serializable_chunks.append({
                            'content': chunk.page_content,
                            'metadata': chunk.metadata
                        })
                    else:
                        serializable_chunks.append(chunk)
                serializable_result['chunks'] = serializable_chunks
            serializable_results.append(serializable_result)
        
        report = {
            'pipeline_version': '2.0.0',
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': asdict(self.config),
            'statistics': self.stats,
            'results': serializable_results,
            'summary': {
                'total_documents': len(results),
                'successful_documents': sum(1 for r in results if r.get('status') == ProcessingStatus.COMPLETED or r.get('status') == 'completed'),
                'failed_documents': sum(1 for r in results if r.get('status') == ProcessingStatus.FAILED or r.get('status') == 'failed'),
                'total_chunks': sum(len(r.get('chunks', [])) for r in results),
                'processing_time': self.stats['processing_time']
            }
        }
        
        report_file = self.output_dir / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        os.chmod(report_file, 0o644)
        
        logger.info(f"Generated processing report: {report_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.stats.copy()
    
    def validate_output(self) -> bool:
        """Validate that all expected output files exist and are readable."""
        required_files = [
            "all_chunks.json",
            "chunk_index.json", 
            "processing_report.json"
        ]
        
        for filename in required_files:
            file_path = self.output_dir / filename
            if not file_path.exists() or not file_path.is_file():
                logger.error(f"Required output file missing: {filename}")
                return False
        
        # Check chunks directory
        chunks_dir = self.output_dir / "chunks"
        if not chunks_dir.exists() or not chunks_dir.is_dir():
            logger.error("Chunks directory missing")
            return False
        
        logger.info("Output validation successful")
        return True

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

def run_integration_tests() -> bool:
    """Run comprehensive integration tests."""
    logger.info("Running integration tests...")
    
    try:
        # Test configuration validation
        config = ProcessingConfig(chunk_size=500, chunk_overlap=100)
        pipeline = DocumentIngestionPipeline(config)
        
        # Test URL validation
        test_url = "https://docs.google.com/document/d/test123/edit"
        assert validate_url(test_url), "URL validation failed"
        
        # Test filename sanitization
        test_filename = "test<>file.txt"
        sanitized = sanitize_filename(test_filename)
        assert sanitized == "test__file.txt", "Filename sanitization failed"
        
        # Test checksum generation
        test_content = "Hello, World!"
        checksum = generate_checksum(test_content)
        assert len(checksum) == 64, "Checksum generation failed"
        
        logger.info("All integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Integration tests failed: {e}")
        return False

def validate_environment() -> bool:
    """Validate that the environment is properly configured."""
    logger.info("Validating environment...")
    
    # Check required environment variables
    required_vars = []
    optional_vars = ['GOOGLE_CREDENTIALS_FILE', 'GOOGLE_TOKEN_FILE']
    
    for var in required_vars:
        if not os.getenv(var):
            logger.error(f"Required environment variable missing: {var}")
            return False
    
    # Check file permissions
    output_dir = Path("source_documents")
    if not output_dir.exists():
        try:
            output_dir.mkdir(exist_ok=True, mode=0o755)
        except Exception as e:
            logger.error(f"Cannot create output directory: {e}")
            return False
    
    logger.info("Environment validation successful")
    return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Enterprise-grade main function with comprehensive error handling."""
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            return 1
        
        # Run integration tests
        if not run_integration_tests():
            logger.error("Integration tests failed")
            return 1
        
        # Initialize pipeline with custom configuration
        config = ProcessingConfig(
            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200')),
            timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', '30')),
            output_dir=os.getenv('OUTPUT_DIR', 'source_documents'),
            enable_checksums=os.getenv('ENABLE_CHECKSUMS', 'true').lower() == 'true'
        )
        
        # Enable embeddings by default, can be disabled via environment variable
        enable_embeddings = os.getenv('ENABLE_EMBEDDINGS', 'true').lower() == 'true'
        
        pipeline = DocumentIngestionPipeline(config, enable_embeddings=enable_embeddings)
        
        # Process documents
        results = pipeline.process_documents()
        
        # Validate output
        if not pipeline.validate_output():
            logger.error("Output validation failed")
            return 1
        
        # Display results
        _display_results(results, pipeline.output_dir)
        
        logger.info("Document ingestion pipeline completed successfully!")
        return 0
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except DocumentIngestionError as e:
        logger.error(f"Document ingestion error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

def _display_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Display comprehensive results summary."""
    stats = results['stats']
    
    print("\n" + "="*80)
    print("🚀 ENTERPRISE DOCUMENT INGESTION PIPELINE - COMPLETED!")
    print("="*80)
    
    print(f"📊 Processing Statistics:")
    print(f"   • Total Documents: {stats['total_documents']}")
    print(f"   • Successfully Processed: {stats['processed_documents']}")
    print(f"   • Failed Documents: {stats['failed_documents']}")
    print(f"   • Total Chunks Generated: {stats['total_chunks']}")
    print(f"   • Processing Time: {stats['processing_time']:.2f} seconds")
    
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"📄 Generated Files:")
    print(f"   • Individual document files (.txt)")
    print(f"   • Document metadata files (_metadata.json)")
    print(f"   • Combined chunks file (all_chunks.json)")
    print(f"   • Embeddings file (embeddings.json)")
    print(f"   • Chunk index file (chunk_index.json)")
    print(f"   • Processing report (processing_report.json)")
    print(f"   • Individual chunk files (chunks/ directory)")
    
    print(f"\n🔧 Configuration Used:")
    print(f"   • Chunk Size: {results.get('config', {}).get('chunk_size', 'N/A')} characters")
    print(f"   • Chunk Overlap: {results.get('config', {}).get('chunk_overlap', 'N/A')} characters")
    print(f"   • Checksums Enabled: {results.get('config', {}).get('enable_checksums', 'N/A')}")
    print(f"   • Embeddings Enabled: {results.get('config', {}).get('enable_embeddings', 'N/A')}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review the processing report: {output_dir}/processing_report.json")
    print(f"   2. Validate chunk quality in: {output_dir}/chunks/")
    print(f"   3. Load chunks for RAG: {output_dir}/all_chunks.json")
    print(f"   4. Integrate with your Discord bot RAG service")
    
    print("="*80)

# =============================================================================
# CLI INTERFACE
# =============================================================================

def cli():
    """Command-line interface for the document ingestion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enterprise Document Ingestion Pipeline for Discord RAG Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py                    # Run with default configuration
  python ingest.py --chunk-size 1500  # Custom chunk size
  python ingest.py --test-only        # Run tests only
  python ingest.py --validate         # Validate environment only
        """
    )
    
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Size of each chunk in characters (default: 1000)')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                       help='Overlap between chunks in characters (default: 200)')
    parser.add_argument('--output-dir', type=str, default='source_documents',
                       help='Output directory for processed documents (default: source_documents)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout for downloads in seconds (default: 30)')
    parser.add_argument('--no-checksums', action='store_true',
                       help='Disable checksum generation')
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Disable embedding generation')
    parser.add_argument('--test-only', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('--validate', action='store_true',
                       help='Validate environment only')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set environment variables from CLI args
    os.environ['CHUNK_SIZE'] = str(args.chunk_size)
    os.environ['CHUNK_OVERLAP'] = str(args.chunk_overlap)
    os.environ['OUTPUT_DIR'] = args.output_dir
    os.environ['TIMEOUT_SECONDS'] = str(args.timeout)
    os.environ['ENABLE_CHECKSUMS'] = str(not args.no_checksums)
    os.environ['ENABLE_EMBEDDINGS'] = str(not args.no_embeddings)
    
    if args.test_only:
        return 0 if run_integration_tests() else 1
    elif args.validate:
        return 0 if validate_environment() else 1
    else:
        return main()

if __name__ == "__main__":
    import sys
    sys.exit(cli())
