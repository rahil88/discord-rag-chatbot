#!/usr/bin/env python3
"""
Azure AI Foundry Client for DeepSeek Model Integration.

This module provides a client for interacting with DeepSeek models
deployed on Azure AI Foundry using the official Azure AI SDK.

Author: AI Assistant
Version: 1.0.0
License: MIT
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import AzureError
except ImportError:
    ChatCompletionsClient = None
    AzureKeyCredential = None
    AzureError = Exception
    SystemMessage = None
    UserMessage = None
    AssistantMessage = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeepSeekResponse:
    """Represents a response from the DeepSeek model."""
    content: str
    usage: Dict[str, Any]
    model: str
    finish_reason: str

class AzureAIFoundryClient:
    """Client for interacting with DeepSeek models on Azure AI Foundry."""
    
    def __init__(self, 
                 endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model_name: Optional[str] = None):
        """
        Initialize the Azure AI Foundry client.
        
        Args:
            endpoint: Azure AI Foundry endpoint URL
            api_key: Azure AI Foundry API key
            model_name: Name of the DeepSeek model to use
        """
        # Check if Azure AI SDK is available
        if ChatCompletionsClient is None:
            raise ImportError(
                "Azure AI SDK not found. Please install it with: "
                "pip install azure-ai-inference azure-identity"
            )
        
        self.endpoint = endpoint or os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_AI_FOUNDRY_API_KEY")
        self.model_name = model_name or os.getenv("AZURE_AI_FOUNDRY_MODEL_NAME", "DeepSeek-R1")
        
        if not self.endpoint:
            raise ValueError("Azure AI Foundry endpoint is required. Set AZURE_AI_FOUNDRY_ENDPOINT environment variable.")
        
        if not self.api_key:
            raise ValueError("Azure AI Foundry API key is required. Set AZURE_AI_FOUNDRY_API_KEY environment variable.")
        
        # Initialize the Azure AI client
        try:
            # Clean endpoint to base URL format as per Azure AI Foundry docs
            base_url = self.endpoint
            if '/models/chat/completions' in base_url:
                base_url = base_url.split('/models/chat/completions')[0]
            elif not base_url.endswith('/models'):
                base_url = base_url.rstrip('/') + '/models'
            
            # Ensure we have the correct base URL format
            if not base_url.endswith('/models'):
                base_url = base_url.rstrip('/') + '/models'
            
            credential = AzureKeyCredential(self.api_key)
            self.client = ChatCompletionsClient(
                endpoint=base_url,
                credential=credential,
                api_version="2024-05-01-preview"
            )
            logger.info(f"Azure AI Foundry client initialized with base URL: {base_url}")
            logger.info(f"Original endpoint: {self.endpoint}")
            logger.info(f"Using model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure AI client: {e}")
            raise
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         max_tokens: int = 1000,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> DeepSeekResponse:
        """
        Generate a response using the DeepSeek model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            
        Returns:
            DeepSeekResponse object with the generated content
        """
        try:
            logger.info(f"Sending request to DeepSeek model: {self.model_name}")
            logger.debug(f"Messages: {messages}")
            
            # Convert messages to Azure AI format
            azure_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    azure_messages.append(SystemMessage(content=content))
                elif role == 'user':
                    azure_messages.append(UserMessage(content=content))
                elif role == 'assistant':
                    azure_messages.append(AssistantMessage(content=content))
                else:
                    # Default to user message
                    azure_messages.append(UserMessage(content=content))

            # Create the chat completions request using Azure AI SDK
            response = self.client.complete(
                messages=azure_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                model=self.model_name
            )
            
            logger.info("Successfully received response from DeepSeek model")
            
            # Extract the response content
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                content = choice.message.content if choice.message else ""
                finish_reason = choice.finish_reason or "unknown"
            else:
                content = "No response generated"
                finish_reason = "no_choice"
            
            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            return DeepSeekResponse(
                content=content,
                usage=usage,
                model=self.model_name,
                finish_reason=finish_reason
            )
            
        except AzureError as e:
            logger.error(f"Azure AI request failed: {e}")
            raise Exception(f"Failed to generate response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def generate_rag_response(self, 
                            query: str, 
                            context_chunks: List[str],
                            max_tokens: int = 1000,
                            temperature: float = 0.7,
                            include_sources: bool = True) -> DeepSeekResponse:
        """
        Generate a RAG response using retrieved context chunks with enhanced prompt construction.
        
        Args:
            query: The user's question
            context_chunks: List of relevant context chunks
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            include_sources: Whether to include source information in the response
            
        Returns:
            DeepSeekResponse object with the generated answer
        """
        # Enhanced context preparation with better formatting
        formatted_context = self._format_context_chunks(context_chunks)
        
        # Create an enhanced system prompt for RAG
        # EXPERIMENT: Change this line to test different prompts:
        # system_prompt = self._create_enhanced_system_prompt()      # Current concise
        # system_prompt = self._create_conversational_system_prompt() # Conversational
        system_prompt = self._create_expert_system_prompt()         # Expert level
        # system_prompt = self._create_enhanced_system_prompt()
        
        # Create the user message with structured context and query
        user_message = self._create_enhanced_user_prompt(query, formatted_context, include_sources)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        logger.info(f"Making RAG API call with {len(context_chunks)} context chunks")
        logger.debug(f"Query: {query}")
        logger.debug(f"Context length: {len(formatted_context)} characters")
        
        return self.generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    def _format_context_chunks(self, context_chunks: List[str]) -> str:
        """
        Format context chunks for better LLM understanding.
        
        Args:
            context_chunks: List of context chunk strings
            
        Returns:
            Formatted context string
        """
        if not context_chunks:
            return "No context information available."
        
        formatted_chunks = []
        for i, chunk in enumerate(context_chunks, 1):
            # Clean and format each chunk
            cleaned_chunk = chunk.strip()
            if cleaned_chunk:
                # Add chunk numbering and formatting
                formatted_chunks.append(f"[Context {i}]\n{cleaned_chunk}")
        
        return "\n\n".join(formatted_chunks)
    
    def _create_enhanced_system_prompt(self) -> str:
        """
        Create an enhanced system prompt for RAG responses.
        
        Returns:
            Enhanced system prompt string
        """
        # EXPERIMENT: Try a more concise prompt
        return """You are a helpful AI assistant that answers questions using provided context documents.

Instructions:
- Base your answer on the provided context
- Be accurate and specific
- If context is insufficient, say so clearly
- Use clear structure and examples when helpful
- Be conversational but professional"""
    
    def _create_conversational_system_prompt(self) -> str:
        """Alternative conversational prompt for experimentation."""
        return """You're a knowledgeable friend who happens to have access to some great documents. Help answer questions using what you know from those documents.

Be helpful, friendly, and conversational while staying accurate. If you're not sure about something based on the documents, just say so. Use examples and analogies when they help explain things better."""
    
    def _create_expert_system_prompt(self) -> str:
        """Alternative expert-level prompt for experimentation."""
        return """You are a domain expert with deep knowledge in the subject matter. Your responses should demonstrate expertise while remaining accessible.

EXPERTISE GUIDELINES:
- Draw connections between different pieces of information
- Provide nuanced understanding beyond surface-level facts
- Explain underlying principles and concepts
- Offer insights that go beyond what's explicitly stated
- Maintain intellectual rigor while being practical"""
    
    def _create_enhanced_user_prompt(self, query: str, formatted_context: str, include_sources: bool = True) -> str:
        """
        Create an enhanced user prompt with structured context and query.
        
        Args:
            query: The user's question
            formatted_context: Formatted context information
            include_sources: Whether to request source information
            
        Returns:
            Enhanced user prompt string
        """
        source_instruction = ""
        if include_sources:
            source_instruction = "\n\nIMPORTANT: At the end of your response, briefly mention which context sections were most relevant to your answer."
        
        # EXPERIMENT: Try a simpler user prompt
        return f"""Context:
{formatted_context}

Question: {query}

Please answer the question using the context above. If the context doesn't contain enough information, let me know."""
    
    def test_connection(self) -> bool:
        """
        Test the connection to Azure AI Foundry.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            test_messages = [
                {"role": "user", "content": "Hello, can you respond with 'Connection test successful'?"}
            ]
            
            response = self.generate_response(
                messages=test_messages,
                max_tokens=50,
                temperature=0.1
            )
            
            logger.info("Connection test successful")
            logger.info(f"Test response: {response.content}")
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


def create_azure_ai_client() -> AzureAIFoundryClient:
    """
    Factory function to create an Azure AI Foundry client.
    
    Returns:
        AzureAIFoundryClient instance
    """
    return AzureAIFoundryClient()


# Example usage and testing
if __name__ == "__main__":
    try:
        # Test the client
        client = create_azure_ai_client()
        
        # Test connection
        if client.test_connection():
            print("✅ Azure AI Foundry connection test successful!")
        else:
            print("❌ Azure AI Foundry connection test failed!")
            exit(1)
        
        # Test basic generation
        test_messages = [
            {"role": "user", "content": "What is artificial intelligence?"}
        ]
        
        response = client.generate_response(test_messages)
        print(f"\nTest Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nPlease install the required packages:")
        print("pip install azure-ai-inference azure-identity")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to set the following environment variables:")
        print("- AZURE_AI_FOUNDRY_ENDPOINT")
        print("- AZURE_AI_FOUNDRY_API_KEY")
        print("- AZURE_AI_FOUNDRY_MODEL_NAME (optional, defaults to DeepSeek-R1)")
