"""
Embedding generation module for creating vector embeddings from text chunks.

This module provides functions to generate embeddings using sentence-transformers
or other embedding models.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Please install it.")


class EmbeddingError(Exception):
    """Custom exception for embedding errors."""
    pass


class EmbeddingGenerator:
    """Class for generating text embeddings."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize EmbeddingGenerator.
        
        Args:
            model_name: Name of the embedding model to use.
            device: Device to use ('cpu', 'cuda', or None for auto).
            batch_size: Batch size for embedding generation.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Please install it: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            logger.info(f"Using device: {device}")
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            error_msg = f"Error loading embedding model: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    def generate_embeddings(
        self,
        texts: Union[List[str], pd.Series],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List or Series of texts to embed.
            show_progress: Whether to show progress bar.
        
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim).
        
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        if not texts:
            raise EmbeddingError("Empty text list provided")
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
        
        except Exception as e:
            error_msg = f"Error generating embeddings: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    def generate_embeddings_for_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'chunk_text',
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for texts in a dataframe.
        
        Args:
            df: Dataframe with text column.
            text_column: Name of column containing texts.
            show_progress: Whether to show progress bar.
        
        Returns:
            Numpy array of embeddings.
        """
        if text_column not in df.columns:
            raise EmbeddingError(f"Text column '{text_column}' not found in dataframe")
        
        texts = df[text_column].tolist()
        return self.generate_embeddings(texts, show_progress=show_progress)


def get_model_info(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> dict:
    """
    Get information about an embedding model.
    
    Args:
        model_name: Name of the model.
    
    Returns:
        Dictionary with model information.
    """
    info = {
        'model_name': model_name,
        'description': '',
        'embedding_dimension': None,
        'max_sequence_length': None
    }
    
    if model_name == "sentence-transformers/all-MiniLM-L6-v2":
        info['description'] = (
            "A lightweight, fast sentence transformer model. "
            "Good balance between speed and quality. "
            "384-dimensional embeddings. "
            "Trained on a large corpus of text pairs."
        )
        info['embedding_dimension'] = 384
        info['max_sequence_length'] = 256
    
    return info

