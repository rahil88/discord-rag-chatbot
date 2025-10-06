"""
Configuration management for FastAPI application
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    environment: str = "development"
    
    # Database settings
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "discord_rag_chatbot"
    
    # AI/ML settings
    openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    
    # Discord settings
    discord_token: Optional[str] = None
    discord_public_key: Optional[str] = None
    discord_application_id: Optional[str] = None
    
    # RAG settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    similarity_threshold: float = 0.7
    max_results: int = 5
    
    # Security settings
    secret_key: str = "your-secret-key-change-in-production"
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings
