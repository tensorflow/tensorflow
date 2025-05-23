# ===================================================================================
# Project: ChatTensorFlow
# File: app/retriever/retriever.py
# Description: This file Loads the persisted vectorDB and returns a retriever instance.
#              Defaults to ChromaDB Instance.
# Author: LALAN KUMAR
# Created: [14-05-2025]
# Updated: [14-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
import sys
from typing import Optional
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore

# Add project root to path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import settings
from app.core.embeddings import get_embeddings
from processor.logger import logging

def get_vector_store(
    persist_directory: Optional[str] = None,
    embedding_provider: Optional[str] = None
) -> VectorStore:
    """Get a vector store instance with the specified embeddings.

    If persist_directory is not provided, it is dynamically selected based on the embedding provider.

    Args:
        persist_directory: Directory to persist the vector store.
        embedding_provider: The embedding provider to use. If None, uses the default from settings.

    Returns:
        A vector store instance.
    """
    # Use the embedding provider from the function argument or from settings
    provider = embedding_provider or settings.EMBEDDING_PROVIDER
    
    # Dynamically determine directory based on the provider
    if persist_directory is None:
        persist_directory = f"DATA/tensorflow_chroma_{provider.lower()}"
    
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Get the appropriate embeddings based on the provider
    embeddings = get_embeddings(provider)
    
    # Create and return the vector store
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def get_retriever(
    persist_directory: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    search_kwargs: Optional[dict] = None
):
    """Initialize and return a retriever.

    Args:
        persist_directory: Directory to persist the vector store. Dynamically derived if not provided.
        embedding_provider: The embedding provider to use. If None, uses the default from settings.
        search_kwargs: Additional search parameters to pass to the retriever.

    Returns:
        A retriever instance.
    """
    search_kwargs = search_kwargs or {"k": 5}
    vector_store = get_vector_store(persist_directory, embedding_provider)
    
    # Check if the vector store has documents
    collection_count = vector_store._collection.count()
    logging.info(f"VectorDB collection count is {collection_count}")
    if collection_count == 0:
        print(f"Warning: Vector store is empty for provider {embedding_provider or settings.EMBEDDING_PROVIDER}")
    
    return vector_store.as_retriever(search_kwargs=search_kwargs)

#get_retriever()