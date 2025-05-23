# ===================================================================================
# Project: ChatTensorFlow
# File: app/core/llm.py
# Description: This file initializes and returns a language model instance based on the specified provider. 
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
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings

def get_llm(provider: Optional[str] = None, streaming: bool = False, callbacks: list = None) -> BaseChatModel:
    """Initialize and return LLM instance based on provider.
    
    Args:
        provider: The LLM provider to use. If None, uses the default from settings.
        
    Returns:
        A LangChain chat model instance.
        
    Raises:
        ValueError: If the requested provider is not supported.
    """
    # Use default provider if none specified
    if provider is None:
        provider = settings.LLM_PROVIDER
    
    # Return the appropriate LLM based on provider
    if provider == settings.LLM_PROVIDER_GEMINI:
        if not settings.GEMINI_API_KEY:
            raise ValueError("Gemini API key not found in environment")
        return ChatGoogleGenerativeAI(model=settings.GEMINI_LLM_MODEL, streaming=streaming, callbacks=callbacks or [])
    
    elif provider == settings.LLM_PROVIDER_OPENAI:
        if not settings.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment")
        return ChatOpenAI(model=settings.OPENAI_LLM_MODEL, streaming=streaming, callbacks=callbacks or [])
    
    elif provider == settings.LLM_PROVIDER_ANTHROPIC:
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key not found in environment")
        return ChatAnthropic(model=settings.ANTHROPIC_LLM_MODEL, streaming=streaming, callbacks=callbacks or [])
    
    elif provider == settings.LLM_PROVIDER_COHERE:
        if not settings.COHERE_API_KEY:
            raise ValueError("Cohere API key not found in environment")
        return ChatCohere(model=settings.COHERE_LLM_MODEL, streaming=streaming, callbacks=callbacks or [])
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    
    
#print("LLM initialized successfully.")