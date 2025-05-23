# ===================================================================================
# Project: ChatTensorFlow
# File: processor/data_ingestion.py
# Description: This file Loads the data from URLs and store it in a VectorStoreDB (Defaults to CHROMA)
# Author: LALAN KUMAR
# Created: [13-05-2025]
# Updated: [13-05-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================
# CAUTION: DO NOT RUN this file until you want to make changes to the vectorStoreDB.
# RECOMMENDATION: It is recommended to run this file multiple times and persist the vectorDB using all the embedding providers.
#                 Currently it persists the vectorDB locally which is efiicient for ChatSklearn application types.
#                 However you can modify the script and persist the vectorDB externally to third party databases.

import os
import sys
import json
import glob
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from langchain.schema import Document
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)


from logger import logging
from config import settings
from app.core.embeddings import get_embeddings


def load_tensorflow_docs(docs_dir: str = "tensorflow_docs") -> List[Dict[str, Any]]:
    """Load TensorFlow documentation from JSON files."""
    logging.info(f"Loading TensorFlow documentation from {docs_dir}...")
    
    # First try to load the combined RAG file
    combined_path = os.path.join(docs_dir, "tensorflow_docs_rag.json")
    if os.path.exists(combined_path):
        logging.info(f"Found combined RAG file: {combined_path}")
        with open(combined_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # If combined file doesn't exist, load individual files
    logging.info("Combined file not found. Loading individual JSON files...")
    json_files = glob.glob(os.path.join(docs_dir, "*.json"))
    
    documents = []
    for file_path in tqdm(json_files, desc="Loading files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                documents.append(doc)
        except Exception as e:
            logging.error(f"Error loading {os.path.basename(file_path)}: {e}")
    
    logging.info(f"Loaded {len(documents)} TensorFlow documentation files")
    return documents

def convert_to_langchain_docs(raw_docs: List[Dict[str, Any]]) -> List[Document]:
    """Convert raw JSON documents to LangChain Document objects."""
    logging.info("Converting to LangChain documents...")
    
    documents = []
    for doc in tqdm(raw_docs, desc="Processing documents"):
        # Skip empty documents
        content = doc.get("content", "")
        if not content:
            continue
            
        # Create LangChain Document with metadata
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "crawled_at": doc.get("crawled_at", "")
                }
            )
        )
    
    logging.info(f"Converted {len(documents)} documents")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for better embedding."""
    logging.info("Splitting documents into chunks...")
    
    # Use MarkdownTextSplitter since TensorFlow docs are in Markdown format
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    logging.info(f"Split into {len(chunks)} chunks")
    return chunks

def persist_vector_db(docs, embeddings, persist_directory):
    """Save documents to Chroma vector store."""
    logging.info(f"Creating vector database at {persist_directory}...")
    
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create and persist the vector database
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Explicitly persist to ensure all data is saved
    logging.info(f"Vector database successfully persisted at: {persist_directory}")
    
    return vectordb

def main(embedding_provider=None):
    """Main function to ingest TensorFlow documentation data."""
    # Use the specified provider or default from settings
    provider = embedding_provider or settings.EMBEDDING_PROVIDER
    logging.info(f"Using embedding provider: {provider}")
    
    # 1. Load TensorFlow documentation from JSON files
    raw_docs = load_tensorflow_docs()
    
    # 2. Convert to LangChain documents
    documents = convert_to_langchain_docs(raw_docs)
    
    # 3. Split into chunks
    chunks = split_documents(documents)
    
    # 4. Get embedding model based on provider
    embeddings = get_embeddings(provider)
    logging.info(f"Using embedding model: {type(embeddings).__name__}")
    
    # 5. Set output directory based on provider name
    persist_directory = f"DATA/tensorflow_chroma_{provider.lower()}"
    
    # 6. Create and persist vector database
    logging.info(f"Persisting vector database (this may take a while)...")
    persist_vector_db(chunks, embeddings, persist_directory)
    
    logging.info(f"✅ TensorFlow documentation successfully embedded and stored at {persist_directory}")

def ingest_all_providers():
    """Ingest data for all available providers."""
    providers = [
        settings.EMBEDDING_PROVIDER_GEMINI,
        settings.EMBEDDING_PROVIDER_OPENAI,
        settings.EMBEDDING_PROVIDER_COHERE
    ]
    
    for provider in providers:
        # Check if API key exists for this provider
        key_var_name = f"{provider.upper()}_API_KEY"
        key = getattr(settings, key_var_name, None) if hasattr(settings, key_var_name) else os.getenv(key_var_name)
        
        if key:
            logging.info(f"Processing data for {provider} provider")
            try:
                main(provider)
                logging.info(f"Successfully processed data for {provider}")
            except Exception as e:
                logging.error(f"Error processing data for {provider}: {str(e)}")
        else:
            logging.warning(f"Skipping {provider} - API key not found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest TensorFlow documentation for vector stores")
    parser.add_argument("--provider", type=str, 
                        help="Specific embedding provider to use (gemini, openai, cohere)")
    parser.add_argument("--all", action="store_true", 
                        help="Process data for all providers with valid API keys")
    parser.add_argument("--docs-dir", type=str, default="tensorflow_docs",
                        help="Directory containing TensorFlow documentation JSON files")
    
    args = parser.parse_args()
    
    if args.all:
        ingest_all_providers()
    elif args.provider:
        # Convert provider name to the constant from settings
        provider_map = {
            "gemini": settings.EMBEDDING_PROVIDER_GEMINI,
            "openai": settings.EMBEDDING_PROVIDER_OPENAI,
            "cohere": settings.EMBEDDING_PROVIDER_COHERE
        }
        
        provider = provider_map.get(args.provider.lower())
        if provider:
            main(provider)
        else:
            logging.error(f"Unknown provider: {args.provider}")
    else:
        # Use default provider from settings
        main()
        
#Example usage:
# To ingest data for all providers with valid API keys:
#python processor/data_ingestion.py --all

# Or for a specific provider:
#python processor/data_ingestion.py --provider gemini
#python processor/data_ingestion.py --provider openai
#python processor/data_ingestion.py --provider cohere
