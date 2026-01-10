"""
Retriever module for RAG system.

This module implements the retrieval component that embeds user questions
and performs similarity search against the vector store.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.embedding_generator import EmbeddingGenerator, EmbeddingError
from src.vector_store_manager import (
    load_vector_store,
    FAISSVectorStore,
    VectorStoreError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Custom exception for retriever errors."""
    pass


class RAGRetriever:
    """Retriever for RAG system that performs semantic search."""
    
    def __init__(
        self,
        vector_store_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type: str = "faiss"
    ):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store_path: Path to the vector store directory.
            embedding_model: Name of the embedding model (must match the one used for indexing).
            vector_store_type: Type of vector store ('faiss' or 'chromadb').
        
        Raises:
            RetrieverError: If initialization fails.
        """
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model_name = embedding_model
        self.vector_store_type = vector_store_type
        
        try:
            # Load vector store
            logger.info(f"Loading vector store from {vector_store_path}")
            self.vector_store = load_vector_store(
                self.vector_store_path,
                store_type=vector_store_type
            )
            logger.info(f"Vector store loaded successfully")
            
            # Initialize embedding generator
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_generator = EmbeddingGenerator(
                model_name=embedding_model,
                batch_size=1  # Single query at a time
            )
            logger.info("Retriever initialized successfully")
            
        except VectorStoreError as e:
            error_msg = f"Error loading vector store: {str(e)}"
            logger.error(error_msg)
            raise RetrieverError(error_msg) from e
        except EmbeddingError as e:
            error_msg = f"Error loading embedding model: {str(e)}"
            logger.error(error_msg)
            raise RetrieverError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing retriever: {str(e)}"
            logger.error(error_msg)
            raise RetrieverError(error_msg) from e
    
    def retrieve(
        self,
        question: str,
        k: int = 5
    ) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve top-k most relevant chunks for a question.
        
        Args:
            question: User's question as a string.
            k: Number of chunks to retrieve (default: 5).
        
        Returns:
            Tuple of (retrieved_chunks, similarity_scores):
            - retrieved_chunks: List of metadata dictionaries for retrieved chunks
            - similarity_scores: List of similarity scores (distances)
        
        Raises:
            RetrieverError: If retrieval fails.
        """
        if not question or not question.strip():
            raise RetrieverError("Question cannot be empty")
        
        try:
            # Embed the question
            logger.debug(f"Embedding question: {question[:100]}...")
            query_embedding = self.embedding_generator.generate_embeddings(
                [question],
                show_progress=False
            )
            
            # Perform similarity search
            logger.debug(f"Searching for top-{k} similar chunks")
            
            if isinstance(self.vector_store, FAISSVectorStore):
                # FAISS returns distances and indices
                distances, indices, metadata_lists = self.vector_store.search(
                    query_embedding,
                    k=k
                )
                
                # Convert distances to similarity scores (lower distance = higher similarity)
                # For L2 distance, we can use 1 / (1 + distance) as similarity
                # For cosine similarity (IP), distances are already similarities
                if self.vector_store.index_type == "L2":
                    # Convert L2 distance to similarity (inverse relationship)
                    similarities = [1.0 / (1.0 + float(dist)) for dist in distances[0]]
                else:
                    # For inner product, higher is better
                    similarities = [float(dist) for dist in distances[0]]
                
                retrieved_chunks = metadata_lists[0] if metadata_lists else []
                
            else:
                # ChromaDB
                results = self.vector_store.search(query_embedding, k=k)
                retrieved_chunks = []
                similarities = []
                
                if results and 'metadatas' in results and 'distances' in results:
                    for i, metadata in enumerate(results['metadatas'][0]):
                        retrieved_chunks.append(metadata)
                        # ChromaDB returns distances (lower is better)
                        if 'distances' in results:
                            dist = results['distances'][0][i] if results['distances'] else 0.0
                            similarities.append(1.0 / (1.0 + float(dist)))
                        else:
                            similarities.append(1.0)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for question")
            
            return retrieved_chunks, similarities
            
        except Exception as e:
            error_msg = f"Error during retrieval: {str(e)}"
            logger.error(error_msg)
            raise RetrieverError(error_msg) from e
    
    def format_context(self, chunks: List[Dict], max_chunks: int = 5) -> str:
        """
        Format retrieved chunks into a context string for the prompt.
        
        Args:
            chunks: List of chunk metadata dictionaries.
            max_chunks: Maximum number of chunks to include in context.
        
        Returns:
            Formatted context string.
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            chunk_text = chunk.get('chunk_text', '')
            complaint_id = chunk.get('complaint_id', 'N/A')
            product = chunk.get('product', 'N/A')
            issue = chunk.get('issue', 'N/A')
            
            context_parts.append(
                f"[Chunk {i+1}]\n"
                f"Complaint ID: {complaint_id}\n"
                f"Product: {product}\n"
                f"Issue: {issue}\n"
                f"Text: {chunk_text}\n"
            )
        
        return "\n".join(context_parts)

