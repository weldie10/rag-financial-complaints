"""
Stratified sampling module for creating proportional samples across product categories.

This module provides functions to create stratified samples that maintain
proportional representation across all product categories.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SamplingError(Exception):
    """Custom exception for sampling errors."""
    pass


def create_complaint_id(df: pd.DataFrame, id_column: Optional[str] = None) -> pd.DataFrame:
    """
    Create or use existing complaint ID column.
    
    Args:
        df: Input dataframe.
        id_column: Name of existing ID column. If None, creates 'complaint_id'.
    
    Returns:
        Dataframe with complaint_id column.
    """
    df = df.copy()
    
    if id_column and id_column in df.columns:
        df['complaint_id'] = df[id_column]
    else:
        # Create unique IDs
        df['complaint_id'] = range(len(df))
        logger.info("Created complaint_id column")
    
    return df


def stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
    stratify_column: str = 'Product',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample maintaining proportional representation.
    
    Args:
        df: Input dataframe.
        sample_size: Target sample size (10,000-15,000).
        stratify_column: Column to stratify by (default: 'Product').
        random_state: Random seed for reproducibility.
    
    Returns:
        Stratified sample dataframe.
    
    Raises:
        SamplingError: If sampling fails.
    """
    if df is None or df.empty:
        error_msg = "DataFrame is None or empty"
        logger.error(error_msg)
        raise SamplingError(error_msg)
    
    if stratify_column not in df.columns:
        error_msg = f"Stratification column '{stratify_column}' not found in dataframe"
        logger.error(error_msg)
        raise SamplingError(error_msg)
    
    try:
        # Ensure we have complaint_id
        if 'complaint_id' not in df.columns:
            df = create_complaint_id(df)
        
        total_rows = len(df)
        
        if sample_size >= total_rows:
            logger.warning(
                f"Sample size ({sample_size}) >= total rows ({total_rows}). "
                f"Returning full dataset."
            )
            return df.copy()
        
        # Calculate distribution of products
        product_counts = df[stratify_column].value_counts()
        product_proportions = product_counts / total_rows
        
        logger.info(f"Original dataset: {total_rows:,} rows")
        logger.info(f"Product distribution:\n{product_counts}")
        
        # Calculate target sample size per product
        samples_per_product = {}
        for product, proportion in product_proportions.items():
            target = int(sample_size * proportion)
            available = product_counts[product]
            # Don't sample more than available
            samples_per_product[product] = min(target, available)
        
        # Adjust if total is less than sample_size due to rounding
        total_allocated = sum(samples_per_product.values())
        if total_allocated < sample_size:
            # Distribute remaining samples proportionally
            remaining = sample_size - total_allocated
            for product in product_proportions.index:
                if samples_per_product[product] < product_counts[product]:
                    additional = min(remaining, product_counts[product] - samples_per_product[product])
                    samples_per_product[product] += additional
                    remaining -= additional
                    if remaining <= 0:
                        break
        
        # Perform stratified sampling
        sampled_dfs = []
        for product, n_samples in samples_per_product.items():
            product_df = df[df[stratify_column] == product]
            if len(product_df) > n_samples:
                product_sample = product_df.sample(
                    n=n_samples,
                    random_state=random_state
                )
            else:
                product_sample = product_df.copy()
            
            sampled_dfs.append(product_sample)
            logger.info(
                f"Sampled {len(product_sample):,} from {product} "
                f"(target: {n_samples:,}, available: {len(product_df):,})"
            )
        
        # Combine samples
        df_sampled = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the final sample
        df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        logger.info(f"Stratified sample created: {len(df_sampled):,} rows")
        logger.info(f"Sample distribution:\n{df_sampled[stratify_column].value_counts()}")
        
        return df_sampled
    
    except Exception as e:
        error_msg = f"Error in stratified sampling: {str(e)}"
        logger.error(error_msg)
        raise SamplingError(error_msg) from e


def get_sampling_statistics(
    df_original: pd.DataFrame,
    df_sampled: pd.DataFrame,
    stratify_column: str = 'Product'
) -> dict:
    """
    Get statistics comparing original and sampled datasets.
    
    Args:
        df_original: Original dataframe.
        df_sampled: Sampled dataframe.
        stratify_column: Column used for stratification.
    
    Returns:
        Dictionary with sampling statistics.
    """
    original_counts = df_original[stratify_column].value_counts()
    sampled_counts = df_sampled[stratify_column].value_counts()
    
    stats = {
        'original_total': len(df_original),
        'sampled_total': len(df_sampled),
        'sampling_ratio': len(df_sampled) / len(df_original),
        'product_distribution': {}
    }
    
    for product in original_counts.index:
        original_count = original_counts[product]
        sampled_count = sampled_counts.get(product, 0)
        original_prop = original_count / len(df_original)
        sampled_prop = sampled_count / len(df_sampled) if len(df_sampled) > 0 else 0
        
        stats['product_distribution'][product] = {
            'original_count': int(original_count),
            'original_proportion': float(original_prop),
            'sampled_count': int(sampled_count),
            'sampled_proportion': float(sampled_prop),
            'proportion_difference': float(abs(original_prop - sampled_prop))
        }
    
    return stats


def save_sample(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    filename: str = "sampled_complaints.csv"
) -> str:
    """
    Save sampled dataframe to CSV.
    
    Args:
        df: Dataframe to save.
        output_path: Full path to output file. If None, uses default path.
        filename: Filename if output_path is None.
    
    Returns:
        Path to saved file.
    
    Raises:
        SamplingError: If saving fails.
    """
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "data" / "processed" / filename
    
    output_path = Path(output_path)
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving sample to {output_path}")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully saved {len(df)} rows to {output_path}")
        return str(output_path)
    
    except Exception as e:
        error_msg = f"Error saving sample: {str(e)}"
        logger.error(error_msg)
        raise SamplingError(error_msg) from e

