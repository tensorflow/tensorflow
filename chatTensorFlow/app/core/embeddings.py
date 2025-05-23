# ===================================================================================
# Project: ChatTensorFlow
# File: app/core/embeddings.py
# Description: This file initializes and returns a embedding model instance based on the specified provider. 
#              Defaults to GEMINI.
# Author: LALAN KUMAR
# Created: [14-05-2025]
# Updated: [14-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings

def get_embeddings(provider: Optional[str] = None) -> Embeddings:
    """Initialize and return embeddings instance based on provider.
    
    Args:
        provider: The embeddings provider to use. If None, uses the default from settings.
        
    Returns:
        A LangChain embeddings instance.
        
    Raises:
        ValueError: If the requested provider is not supported.
    """
    # Use default provider if none specified
    if provider is None:
        provider = settings.EMBEDDING_PROVIDER
    
    # Return the appropriate embeddings based on provider
    if provider == settings.EMBEDDING_PROVIDER_GEMINI:
        if not settings.GEMINI_API_KEY:
            raise ValueError("Gemini API key not found in environment")
        return GoogleGenerativeAIEmbeddings(model=settings.GEMINI_EMBEDDING_MODEL)
    
    elif provider == settings.EMBEDDING_PROVIDER_OPENAI:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment")
        return OpenAIEmbeddings(model=settings.OPENAI_EMBEDDING_MODEL)
    
    elif provider == settings.EMBEDDING_PROVIDER_COHERE:
        if not settings.COHERE_API_KEY:
            raise ValueError("Cohere API key not found in environment")
        return CohereEmbeddings(model=settings.COHERE_EMBEDDING_MODEL)
    
    else:
        raise ValueError(f"Unsupported embeddings provider: {provider}")
    
#print("Embeddings module loaded successfully.")