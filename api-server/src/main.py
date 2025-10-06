"""
Simple FastAPI Server for RAG Chatbot
Single endpoint: POST /api/ask
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import logging
import time
import structlog
from dotenv import load_dotenv
from src.rag_service import RAGService, create_rag_service, RAGResponse

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = structlog.get_logger(__name__)

# Pydantic models for request/response
class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    min_similarity: float = 0.15

class SourceInfo(BaseModel):
    content: str
    similarity_score: float
    source_document: str
    chunk_id: int

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
    confidence: float = 0.0
    processing_time: float = 0.0
    total_sources_found: int = 0

# Create FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Simple API for asking questions using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses"""
    start_time = time.time()
    
    # Log incoming request
    logger.info(
        "Request received",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown")
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=round(process_time, 4),
        client_ip=request.client.host if request.client else "unknown"
    )
    
    return response

# Initialize RAG service
rag_service: RAGService = None

async def initialize_rag_service():
    """Initialize the RAG service on startup"""
    global rag_service
    try:
        logger.info("Initializing RAG service...")
        
        # Get Azure AI configuration from environment
        azure_endpoint = os.getenv("AZURE_AI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_AI_API_KEY")
        azure_model_name = os.getenv("AZURE_AI_MODEL_NAME", "deepseek-chat")
        
        # Get MongoDB configuration
        mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        use_mongodb = bool(mongodb_connection_string)
        
        # Log configuration (without sensitive data)
        logger.info(
            "RAG service configuration",
            mongodb_enabled=use_mongodb,
            azure_ai_enabled=bool(azure_endpoint and azure_api_key),
            azure_model=azure_model_name,
            mongodb_configured=bool(mongodb_connection_string),
            azure_endpoint_configured=bool(azure_endpoint)
        )
        
        # Create RAG service
        rag_service = create_rag_service(
            use_mongodb=use_mongodb,
            mongodb_connection_string=mongodb_connection_string,
            use_azure_ai=bool(azure_endpoint and azure_api_key),
            azure_endpoint=azure_endpoint,
            azure_api_key=azure_api_key,
            azure_model_name=azure_model_name
        )
        
        logger.info("RAG service initialized successfully!")
        
        # Get document summary
        summary = rag_service.get_document_summary()
        logger.info(
            "Document summary loaded",
            total_chunks=summary['total_chunks'],
            total_documents=summary['total_documents']
        )
        
    except Exception as e:
        logger.error(
            "Failed to initialize RAG service",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"RAG service initialization failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_rag_service()

@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get an intelligent response using RAG
    """
    global rag_service
    
    if rag_service is None:
        logger.error("RAG service not initialized when processing request")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        logger.info(
            "Processing RAG query",
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        # Generate response using RAG service
        rag_response: RAGResponse = rag_service.generate_response(
            query=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity
        )
        
        # Convert RAG response to API response format
        sources = [
            SourceInfo(
                content=source.content,
                similarity_score=source.similarity_score,
                source_document=source.source_document,
                chunk_id=source.chunk_id
            )
            for source in rag_response.sources
        ]
        
        response = AskResponse(
            answer=rag_response.answer,
            sources=sources,
            confidence=rag_response.confidence,
            processing_time=rag_response.processing_time,
            total_sources_found=len(rag_response.sources)
        )
        
        logger.info(
            "Query processed successfully",
            query=request.query,
            processing_time=rag_response.processing_time,
            sources_found=len(rag_response.sources),
            confidence=rag_response.confidence,
            answer_length=len(rag_response.answer)
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "Error processing RAG query",
            query=request.query,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint"""
    global rag_service
    
    rag_status = "initialized" if rag_service is not None else "not_initialized"
    
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "rag_service_status": rag_status,
        "endpoints": {
            "ask": "POST /api/ask",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global rag_service
    
    if rag_service is None:
        logger.warning("Health check failed: RAG service not initialized")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "rag_service": "not_initialized",
                "message": "RAG service not available"
            }
        )
    
    try:
        # Get document summary to verify service is working
        summary = rag_service.get_document_summary()
        
        logger.info(
            "Health check successful",
            rag_service_status="initialized",
            total_chunks=summary["total_chunks"],
            total_documents=summary["total_documents"]
        )
        
        return {
            "status": "healthy",
            "rag_service": "initialized",
            "documents": {
                "total_chunks": summary["total_chunks"],
                "total_documents": summary["total_documents"]
            },
            "timestamp": "2024-01-01T00:00:00Z"  # You can add actual timestamp if needed
        }
        
    except Exception as e:
        logger.error(
            "Health check failed",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "rag_service": "error",
                "message": str(e)
            }
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc):
    """Global exception handler"""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=str(request.url),
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    print(f"🌐 Starting server on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
