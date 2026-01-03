"""
Data preprocessing module for cleaning and filtering complaint narratives.

This module provides functions to clean text, filter data, and prepare
the dataset for the RAG pipeline.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


# Common boilerplate patterns to remove
BOILERPLATE_PATTERNS = [
    r'\bI am writing to file a complaint\b',
    r'\bI am writing to complain\b',
    r'\bThis is a complaint regarding\b',
    r'\bI would like to file a complaint\b',
    r'\bPlease be advised that\b',
    r'\bI am contacting you regarding\b',
]


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_special_chars: bool = True,
    remove_boilerplate: bool = True,
    normalize_whitespace: bool = True
) -> str:
    """
    Clean a single text narrative.
    
    Args:
        text: Input text to clean.
        lowercase: Whether to convert to lowercase.
        remove_special_chars: Whether to remove special characters (keeps alphanumeric and basic punctuation).
        remove_boilerplate: Whether to remove common boilerplate phrases.
        normalize_whitespace: Whether to normalize whitespace.
    
    Returns:
        Cleaned text string.
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Remove boilerplate phrases
    if remove_boilerplate:
        for pattern in BOILERPLATE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
    if remove_special_chars:
        # Keep letters, numbers, spaces, periods, commas, question marks, exclamation marks
        text = re.sub(r'[^a-z0-9\s.,!?]', ' ', text)
    
    # Normalize whitespace
    if normalize_whitespace:
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()  # Remove leading/trailing whitespace
    
    return text


def filter_by_products(
    df: pd.DataFrame,
    products: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filter dataframe to include only specified products.
    
    Args:
        df: Input dataframe.
        products: List of product names to include. If None, uses default products.
    
    Returns:
        Filtered dataframe.
    
    Raises:
        PreprocessingError: If filtering fails.
    """
    if products is None:
        # Default products as specified in the task
        products = [
            'Credit card',
            'Personal loan',
            'Savings account',
            'Money transfers'
        ]
    
    try:
        logger.info(f"Filtering data for products: {products}")
        
        # Normalize product names (handle case sensitivity and whitespace)
        df_filtered = df[df['Product'].str.strip().str.title().isin(
            [p.strip().title() for p in products]
        )].copy()
        
        logger.info(f"Filtered from {len(df)} to {len(df_filtered)} rows")
        
        return df_filtered
    
    except Exception as e:
        error_msg = f"Error filtering by products: {str(e)}"
        logger.error(error_msg)
        raise PreprocessingError(error_msg) from e


def remove_empty_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with empty or missing Consumer complaint narrative.
    
    Args:
        df: Input dataframe.
    
    Returns:
        Dataframe with empty narratives removed.
    
    Raises:
        PreprocessingError: If column doesn't exist or processing fails.
    """
    if 'Consumer complaint narrative' not in df.columns:
        error_msg = "Column 'Consumer complaint narrative' not found in dataframe"
        logger.error(error_msg)
        raise PreprocessingError(error_msg)
    
    try:
        initial_count = len(df)
        
        # Remove rows where narrative is NaN, None, or empty string
        df_filtered = df[
            df['Consumer complaint narrative'].notna() &
            (df['Consumer complaint narrative'].astype(str).str.strip() != '')
        ].copy()
        
        removed_count = initial_count - len(df_filtered)
        logger.info(f"Removed {removed_count} rows with empty narratives "
                   f"({initial_count} -> {len(df_filtered)})")
        
        return df_filtered
    
    except Exception as e:
        error_msg = f"Error removing empty narratives: {str(e)}"
        logger.error(error_msg)
        raise PreprocessingError(error_msg) from e


def preprocess_complaints(
    df: pd.DataFrame,
    products: Optional[List[str]] = None,
    lowercase: bool = True,
    remove_special_chars: bool = True,
    remove_boilerplate: bool = True,
    normalize_whitespace: bool = True
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline for complaint data.
    
    Args:
        df: Input dataframe with complaint data.
        products: List of products to filter by. If None, uses default products.
        lowercase: Whether to lowercase text.
        remove_special_chars: Whether to remove special characters.
        remove_boilerplate: Whether to remove boilerplate text.
        normalize_whitespace: Whether to normalize whitespace.
    
    Returns:
        Preprocessed dataframe.
    
    Raises:
        PreprocessingError: If preprocessing fails.
    """
    try:
        logger.info("Starting preprocessing pipeline")
        
        # Step 1: Filter by products
        df_processed = filter_by_products(df, products)
        
        # Step 2: Remove empty narratives
        df_processed = remove_empty_narratives(df_processed)
        
        # Step 3: Clean text narratives
        logger.info("Cleaning text narratives")
        df_processed['Consumer complaint narrative'] = df_processed[
            'Consumer complaint narrative'
        ].apply(
            lambda x: clean_text(
                x,
                lowercase=lowercase,
                remove_special_chars=remove_special_chars,
                remove_boilerplate=remove_boilerplate,
                normalize_whitespace=normalize_whitespace
            )
        )
        
        # Step 4: Remove any narratives that became empty after cleaning
        df_processed = remove_empty_narratives(df_processed)
        
        logger.info(f"Preprocessing complete. Final dataset: {len(df_processed)} rows")
        
        return df_processed
    
    except Exception as e:
        error_msg = f"Error in preprocessing pipeline: {str(e)}"
        logger.error(error_msg)
        raise PreprocessingError(error_msg) from e


def save_processed_data(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    filename: str = "filtered_complaints.csv"
) -> str:
    """
    Save processed dataframe to CSV file.
    
    Args:
        df: Dataframe to save.
        output_path: Full path to output file. If None, uses default path.
        filename: Filename if output_path is None.
    
    Returns:
        Path to saved file.
    
    Raises:
        PreprocessingError: If saving fails.
    """
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "data" / "processed" / filename
    
    output_path = Path(output_path)
    
    try:
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed data to {output_path}")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Successfully saved {len(df)} rows to {output_path}")
        return str(output_path)
    
    except Exception as e:
        error_msg = f"Error saving processed data: {str(e)}"
        logger.error(error_msg)
        raise PreprocessingError(error_msg) from e

