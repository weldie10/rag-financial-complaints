"""
Vector store management module for FAISS and ChromaDB.

This module provides functions to create, save, and load vector stores
with metadata support.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Please install it: pip install faiss-cpu or faiss-gpu")

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not available. Please install it: pip install chromadb")


class VectorStoreError(Exception):
    """Custom exception for vector store errors."""
    pass


class FAISSVectorStore:
    """FAISS-based vector store with metadata support."""
    
    def __init__(self, embedding_dim: int, index_type: str = "L2"):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings.
            index_type: Type of index ('L2' for L2 distance, 'IP' for inner product).
        """
        if not FAISS_AVAILABLE:
            raise VectorStoreError("FAISS is not installed. Please install it: pip install faiss-cpu")
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(embedding_dim)
        elif index_type == "IP":
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise VectorStoreError(f"Unsupported index type: {index_type}")
        
        # Store metadata for each vector
        self.metadata: List[Dict] = []
        
        logger.info(f"Initialized FAISS index (dim={embedding_dim}, type={index_type})")
    
    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ) -> None:
        """
        Add vectors and metadata to the index.
        
        Args:
            embeddings: Numpy array of embeddings (n_vectors, embedding_dim).
            metadata: List of metadata dictionaries, one per vector.
        
        Raises:
            VectorStoreError: If dimensions don't match or metadata length doesn't match.
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise VectorStoreError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        if len(metadata) != embeddings.shape[0]:
            raise VectorStoreError(
                f"Metadata length mismatch: {len(metadata)} metadata items "
                f"for {embeddings.shape[0]} vectors"
            )
        
        # Normalize for cosine similarity if using IP
        if self.index_type == "IP":
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store metadata
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(metadata)} vectors to index. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
        """
        Search for similar vectors.
        
        Args:
            query_embeddings: Query embeddings (n_queries, embedding_dim).
            k: Number of nearest neighbors to return.
        
        Returns:
            Tuple of (distances, indices, metadata_lists).
            - distances: (n_queries, k) array of distances
            - indices: (n_queries, k) array of indices
            - metadata_lists: List of lists of metadata dicts
        """
        if query_embeddings.shape[1] != self.embedding_dim:
            raise VectorStoreError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {query_embeddings.shape[1]}"
            )
        
        # Normalize for cosine similarity if using IP
        if self.index_type == "IP":
            faiss.normalize_L2(query_embeddings)
        
        distances, indices = self.index.search(
            query_embeddings.astype('float32'),
            min(k, self.index.ntotal)
        )
        
        # Get metadata for each result
        metadata_lists = []
        for query_idx in range(len(query_embeddings)):
            query_metadata = [
                self.metadata[idx] for idx in indices[query_idx]
            ]
            metadata_lists.append(query_metadata)
        
        return distances, indices, metadata_lists
    
    def save(self, directory: Union[str, Path]) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            directory: Directory to save to.
        
        Raises:
            VectorStoreError: If saving fails.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            index_path = directory / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")
            
            # Save metadata
            metadata_path = directory / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved metadata to {metadata_path}")
            
            # Save index info
            info = {
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'num_vectors': self.index.ntotal
            }
            info_path = directory / "index_info.pkl"
            with open(info_path, 'wb') as f:
                pickle.dump(info, f)
            logger.info(f"Saved index info to {info_path}")
        
        except Exception as e:
            error_msg = f"Error saving FAISS index: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> 'FAISSVectorStore':
        """
        Load FAISS index and metadata from disk.
        
        Args:
            directory: Directory to load from.
        
        Returns:
            Loaded FAISSVectorStore instance.
        
        Raises:
            VectorStoreError: If loading fails.
        """
        directory = Path(directory)
        
        try:
            # Load index info
            info_path = directory / "index_info.pkl"
            if not info_path.exists():
                raise VectorStoreError(f"Index info not found: {info_path}")
            
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
            
            # Create instance
            instance = cls(info['embedding_dim'], info['index_type'])
            
            # Load FAISS index
            index_path = directory / "faiss.index"
            if not index_path.exists():
                raise VectorStoreError(f"FAISS index not found: {index_path}")
            
            instance.index = faiss.read_index(str(index_path))
            
            # Load metadata
            metadata_path = directory / "metadata.pkl"
            if not metadata_path.exists():
                raise VectorStoreError(f"Metadata not found: {metadata_path}")
            
            with open(metadata_path, 'rb') as f:
                instance.metadata = pickle.load(f)
            
            logger.info(f"Loaded FAISS index with {instance.index.ntotal} vectors")
            return instance
        
        except Exception as e:
            error_msg = f"Error loading FAISS index: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e


class ChromaDBVectorStore:
    """ChromaDB-based vector store."""
    
    def __init__(
        self,
        collection_name: str = "complaints",
        persist_directory: Optional[Union[str, Path]] = None
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection.
            persist_directory: Directory to persist data. If None, uses in-memory.
        """
        if not CHROMADB_AVAILABLE:
            raise VectorStoreError("ChromaDB is not installed. Please install it: pip install chromadb")
        
        self.collection_name = collection_name
        
        if persist_directory:
            persist_directory = Path(persist_directory)
            persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
    
    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add vectors and metadata to ChromaDB.
        
        Args:
            embeddings: Numpy array of embeddings.
            metadata: List of metadata dictionaries.
            ids: Optional list of IDs. If None, generates sequential IDs.
        """
        if len(metadata) != embeddings.shape[0]:
            raise VectorStoreError(
                f"Metadata length mismatch: {len(metadata)} metadata items "
                f"for {embeddings.shape[0]} vectors"
            )
        
        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(metadata))]
        
        # Convert embeddings to list of lists
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Added {len(metadata)} vectors to ChromaDB collection")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
        n_results: Optional[int] = None
    ) -> Dict:
        """
        Search for similar vectors.
        
        Args:
            query_embeddings: Query embeddings.
            k: Number of results per query.
            n_results: Alias for k (ChromaDB parameter).
        
        Returns:
            Dictionary with results.
        """
        if n_results is None:
            n_results = k
        
        # Convert to list of lists
        query_embeddings_list = query_embeddings.tolist()
        
        results = self.collection.query(
            query_embeddings=query_embeddings_list,
            n_results=n_results
        )
        
        return results


def create_vector_store(
    store_type: str = "faiss",
    embedding_dim: Optional[int] = None,
    **kwargs
) -> Union[FAISSVectorStore, ChromaDBVectorStore]:
    """
    Factory function to create a vector store.
    
    Args:
        store_type: Type of store ('faiss' or 'chromadb').
        embedding_dim: Embedding dimension (required for FAISS).
        **kwargs: Additional arguments for the vector store.
    
    Returns:
        Vector store instance.
    """
    if store_type.lower() == "faiss":
        if embedding_dim is None:
            raise VectorStoreError("embedding_dim is required for FAISS")
        return FAISSVectorStore(embedding_dim, **kwargs)
    elif store_type.lower() == "chromadb":
        return ChromaDBVectorStore(**kwargs)
    else:
        raise VectorStoreError(f"Unsupported store type: {store_type}")


def save_vector_store(
    vector_store: Union[FAISSVectorStore, ChromaDBVectorStore],
    directory: Union[str, Path],
    store_type: str = "faiss"
) -> None:
    """
    Save vector store to disk.
    
    Args:
        vector_store: Vector store instance.
        directory: Directory to save to.
        store_type: Type of store.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    if store_type.lower() == "faiss":
        if isinstance(vector_store, FAISSVectorStore):
            vector_store.save(directory)
        else:
            raise VectorStoreError("Vector store type mismatch")
    elif store_type.lower() == "chromadb":
        # ChromaDB persists automatically if using PersistentClient
        logger.info("ChromaDB collection persisted automatically")
    else:
        raise VectorStoreError(f"Unsupported store type: {store_type}")


def load_vector_store(
    directory: Union[str, Path],
    store_type: str = "faiss",
    **kwargs
) -> Union[FAISSVectorStore, ChromaDBVectorStore]:
    """
    Load vector store from disk.
    
    Args:
        directory: Directory to load from.
        store_type: Type of store.
        **kwargs: Additional arguments for loading.
    
    Returns:
        Loaded vector store instance.
    """
    directory = Path(directory)
    
    if store_type.lower() == "faiss":
        return FAISSVectorStore.load(directory)
    elif store_type.lower() == "chromadb":
        collection_name = kwargs.get('collection_name', 'complaints')
        return ChromaDBVectorStore(collection_name, persist_directory=directory)
    else:
        raise VectorStoreError(f"Unsupported store type: {store_type}")

