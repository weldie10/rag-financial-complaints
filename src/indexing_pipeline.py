"""
Main indexing pipeline script.

This script orchestrates the complete workflow:
1. Load processed data
2. Create stratified sample
3. Chunk texts
4. Generate embeddings
5. Create and save vector store
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.data_loader import load_cfpb_complaints, DataLoadError
from src.stratified_sampler import (
    stratified_sample,
    create_complaint_id,
    get_sampling_statistics,
    save_sample,
    SamplingError
)
from src.text_chunker import TextChunker, analyze_chunking_results, ChunkingError
from src.embedding_generator import EmbeddingGenerator, get_model_info, EmbeddingError
from src.vector_store_manager import (
    create_vector_store,
    save_vector_store,
    VectorStoreError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('indexing.log')
    ]
)
logger = logging.getLogger(__name__)


def load_processed_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load processed complaint data.
    
    Args:
        file_path: Path to processed data. If None, uses default path.
    
    Returns:
        Loaded dataframe.
    """
    if file_path is None:
        project_root = Path(__file__).parent.parent
        file_path = project_root / "data" / "processed" / "filtered_complaints.csv"
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise DataLoadError(f"Processed data file not found: {file_path}")
    
    logger.info(f"Loading processed data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Ensure complaint_id exists
    if 'complaint_id' not in df.columns:
        df = create_complaint_id(df)
    
    logger.info(f"Loaded {len(df)} rows")
    return df


def main(
    sample_size: int = 12000,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    vector_store_type: str = "faiss",
    processed_data_path: Optional[str] = None
):
    """
    Main indexing pipeline.
    
    Args:
        sample_size: Target sample size (10,000-15,000).
        chunk_size: Size of text chunks in characters.
        chunk_overlap: Overlap between chunks in characters.
        embedding_model: Name of embedding model to use.
        vector_store_type: Type of vector store ('faiss' or 'chromadb').
        processed_data_path: Path to processed data. If None, uses default.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting Vector Store Indexing Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load processed data
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Loading Processed Data")
        logger.info("=" * 60)
        
        df = load_processed_data(processed_data_path)
        
        # Step 2: Create stratified sample
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Creating Stratified Sample")
        logger.info("=" * 60)
        
        df_sampled = stratified_sample(
            df,
            sample_size=sample_size,
            stratify_column='Product',
            random_state=42
        )
        
        # Get sampling statistics
        sampling_stats = get_sampling_statistics(df, df_sampled)
        logger.info(f"\nSampling Statistics:")
        logger.info(f"  Original: {sampling_stats['original_total']:,} rows")
        logger.info(f"  Sampled: {sampling_stats['sampled_total']:,} rows")
        logger.info(f"  Sampling ratio: {sampling_stats['sampling_ratio']:.2%}")
        
        # Save sample
        save_sample(df_sampled, filename="sampled_complaints.csv")
        
        # Step 3: Chunk texts
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Chunking Texts")
        logger.info("=" * 60)
        logger.info(f"Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
        
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_langchain=True
        )
        
        # Preserve important metadata columns
        metadata_columns = [
            'complaint_id',
            'Product',
            'Sub-product',
            'Issue',
            'Company',
            'State',
            'Date received'
        ]
        # Only include columns that exist
        metadata_columns = [col for col in metadata_columns if col in df_sampled.columns]
        
        df_chunks = chunker.chunk_dataframe(
            df_sampled,
            text_column='Consumer complaint narrative',
            metadata_columns=metadata_columns
        )
        
        # Analyze chunking results
        chunking_stats = analyze_chunking_results(df_chunks, df_sampled)
        logger.info(f"\nChunking Statistics:")
        logger.info(f"  Original texts: {chunking_stats['original_texts']:,}")
        logger.info(f"  Total chunks: {chunking_stats['total_chunks']:,}")
        logger.info(f"  Avg chunks per text: {chunking_stats['avg_chunks_per_text']:.2f}")
        logger.info(f"  Mean chunk length: {chunking_stats['chunk_length_stats']['mean']:.1f} chars")
        
        # Step 4: Generate embeddings
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Generating Embeddings")
        logger.info("=" * 60)
        logger.info(f"Model: {embedding_model}")
        
        # Get model info
        model_info = get_model_info(embedding_model)
        logger.info(f"  Embedding dimension: {model_info['embedding_dimension']}")
        logger.info(f"  Description: {model_info['description']}")
        
        embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            batch_size=32
        )
        
        embeddings = embedding_generator.generate_embeddings_for_dataframe(
            df_chunks,
            text_column='chunk_text',
            show_progress=True
        )
        
        logger.info(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
        
        # Step 5: Create vector store
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Creating Vector Store")
        logger.info("=" * 60)
        logger.info(f"Store type: {vector_store_type}")
        
        # Prepare metadata
        metadata_list = []
        for _, row in df_chunks.iterrows():
            metadata = {
                'complaint_id': str(row.get('complaint_id', '')),
                'chunk_id': str(row.get('chunk_id', '')),
                'chunk_index': int(row.get('chunk_index', 0)),
                'product': str(row.get('Product', '')),
                'sub_product': str(row.get('Sub-product', '')),
                'issue': str(row.get('Issue', '')),
                'company': str(row.get('Company', '')),
                'state': str(row.get('State', '')),
                'date_received': str(row.get('Date received', '')),
                'num_chunks': int(row.get('num_chunks', 1)),
                'chunk_text': str(row.get('chunk_text', ''))[:500]  # Store first 500 chars for reference
            }
            metadata_list.append(metadata)
        
        # Create vector store
        vector_store = create_vector_store(
            store_type=vector_store_type,
            embedding_dim=embeddings.shape[1]
        )
        
        # Add vectors and metadata
        vector_store.add_vectors(embeddings, metadata_list)
        
        # Step 6: Save vector store
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Saving Vector Store")
        logger.info("=" * 60)
        
        project_root = Path(__file__).parent.parent
        vector_store_dir = project_root / "vector_store"
        
        save_vector_store(vector_store, vector_store_dir, store_type=vector_store_type)
        
        logger.info("\n" + "=" * 60)
        logger.info("INDEXING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Vector store saved to: {vector_store_dir}")
        logger.info(f"Total vectors indexed: {len(embeddings):,}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("INDEXING SUMMARY")
        print("=" * 60)
        print(f"Original dataset: {len(df):,} rows")
        print(f"Sampled dataset: {len(df_sampled):,} rows")
        print(f"Total chunks: {len(df_chunks):,}")
        print(f"Embedding model: {embedding_model}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"Vector store type: {vector_store_type}")
        print(f"Vector store location: {vector_store_dir}")
        print("=" * 60)
        
        return vector_store, df_chunks, embeddings
    
    except DataLoadError as e:
        logger.error(f"Data loading error: {e}")
        sys.exit(1)
    
    except SamplingError as e:
        logger.error(f"Sampling error: {e}")
        sys.exit(1)
    
    except ChunkingError as e:
        logger.error(f"Chunking error: {e}")
        sys.exit(1)
    
    except EmbeddingError as e:
        logger.error(f"Embedding error: {e}")
        sys.exit(1)
    
    except VectorStoreError as e:
        logger.error(f"Vector store error: {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector Store Indexing Pipeline")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12000,
        help="Target sample size (default: 12000)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size in characters (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--vector-store-type",
        type=str,
        default="faiss",
        choices=["faiss", "chromadb"],
        help="Vector store type (default: faiss)"
    )
    parser.add_argument(
        "--processed-data-path",
        type=str,
        default=None,
        help="Path to processed data CSV (default: data/processed/filtered_complaints.csv)"
    )
    
    args = parser.parse_args()
    
    main(
        sample_size=args.sample_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        vector_store_type=args.vector_store_type,
        processed_data_path=args.processed_data_path
    )

