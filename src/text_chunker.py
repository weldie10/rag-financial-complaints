"""
Text chunking module for splitting long narratives into smaller chunks.

This module provides functions to chunk text using various strategies,
including LangChain's RecursiveCharacterTextSplitter.
"""

import logging
from typing import List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Using custom chunker.")


class ChunkingError(Exception):
    """Custom exception for chunking errors."""
    pass


def simple_text_chunk(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separator: str = " "
) -> List[str]:
    """
    Simple text chunking function (fallback if LangChain unavailable).
    
    Args:
        text: Text to chunk.
        chunk_size: Maximum size of each chunk (in characters).
        chunk_overlap: Number of characters to overlap between chunks.
        separator: Separator to use when splitting.
    
    Returns:
        List of text chunks.
    """
    if not text or len(text.strip()) == 0:
        return []
    
    text = text.strip()
    
    # If text is shorter than chunk_size, return as single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break
        
        # Try to break at separator to avoid splitting words
        if separator in text[start:end]:
            # Find last occurrence of separator in chunk
            last_sep = text.rfind(separator, start, end)
            if last_sep > start:
                end = last_sep + len(separator)
        
        chunks.append(text[start:end].strip())
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start < 0:
            start = 0
    
    return chunks


class TextChunker:
    """Class for chunking text narratives."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        use_langchain: bool = True
    ):
        """
        Initialize TextChunker.
        
        Args:
            chunk_size: Maximum size of each chunk (in characters).
            chunk_overlap: Number of characters to overlap between chunks.
            use_langchain: Whether to use LangChain's RecursiveCharacterTextSplitter.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        
        if self.use_langchain:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            logger.info("Using LangChain RecursiveCharacterTextSplitter")
        else:
            logger.info("Using custom text chunker")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk a single text.
        
        Args:
            text: Text to chunk.
        
        Returns:
            List of text chunks.
        """
        if not text or pd.isna(text):
            return []
        
        text = str(text).strip()
        
        if len(text) == 0:
            return []
        
        try:
            if self.use_langchain:
                chunks = self.splitter.split_text(text)
            else:
                chunks = simple_text_chunk(
                    text,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            
            # Filter out empty chunks
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            
            return chunks
        
        except Exception as e:
            error_msg = f"Error chunking text: {str(e)}"
            logger.error(error_msg)
            raise ChunkingError(error_msg) from e
    
    def chunk_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'Consumer complaint narrative',
        metadata_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Chunk all texts in a dataframe and create expanded dataframe with chunks.
        
        Args:
            df: Input dataframe.
            text_column: Name of column containing text to chunk.
            metadata_columns: Columns to preserve as metadata. If None, preserves all.
        
        Returns:
            Dataframe with one row per chunk, including metadata.
        
        Raises:
            ChunkingError: If chunking fails.
        """
        if text_column not in df.columns:
            error_msg = f"Text column '{text_column}' not found in dataframe"
            logger.error(error_msg)
            raise ChunkingError(error_msg)
        
        if metadata_columns is None:
            # Preserve all columns except the text column
            metadata_columns = [col for col in df.columns if col != text_column]
        
        try:
            logger.info(f"Chunking {len(df)} texts with chunk_size={self.chunk_size}, "
                       f"chunk_overlap={self.chunk_overlap}")
            
            chunk_rows = []
            
            for idx, row in df.iterrows():
                text = row[text_column]
                chunks = self.chunk_text(text)
                
                if len(chunks) == 0:
                    # Skip if no chunks created
                    continue
                
                # Create a row for each chunk
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_row = {
                        'chunk_id': f"{row.get('complaint_id', idx)}_{chunk_idx}",
                        'chunk_index': chunk_idx,
                        'chunk_text': chunk_text,
                        'num_chunks': len(chunks),
                        **{col: row[col] for col in metadata_columns}
                    }
                    chunk_rows.append(chunk_row)
            
            df_chunks = pd.DataFrame(chunk_rows)
            
            logger.info(
                f"Created {len(df_chunks)} chunks from {len(df)} texts "
                f"(avg {len(df_chunks)/len(df):.2f} chunks per text)"
            )
            
            return df_chunks
        
        except Exception as e:
            error_msg = f"Error chunking dataframe: {str(e)}"
            logger.error(error_msg)
            raise ChunkingError(error_msg) from e


def analyze_chunking_results(
    df_chunks: pd.DataFrame,
    original_df: pd.DataFrame
) -> dict:
    """
    Analyze chunking results and provide statistics.
    
    Args:
        df_chunks: Dataframe with chunks.
        original_df: Original dataframe before chunking.
    
    Returns:
        Dictionary with chunking statistics.
    """
    stats = {
        'original_texts': len(original_df),
        'total_chunks': len(df_chunks),
        'avg_chunks_per_text': len(df_chunks) / len(original_df) if len(original_df) > 0 else 0,
        'chunk_length_stats': {
            'mean': df_chunks['chunk_text'].str.len().mean(),
            'median': df_chunks['chunk_text'].str.len().median(),
            'min': df_chunks['chunk_text'].str.len().min(),
            'max': df_chunks['chunk_text'].str.len().max(),
            'std': df_chunks['chunk_text'].str.len().std()
        },
        'chunks_per_text_distribution': df_chunks.groupby('complaint_id')['chunk_id'].count().describe().to_dict()
    }
    
    return stats

