"""
Data loading module for CFPB complaint dataset.

This module provides functions to load and validate the CFPB complaint dataset
with proper error handling and logging.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass


def load_cfpb_complaints(
    file_path: Optional[str] = None,
    nrows: Optional[int] = None,
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Load CFPB complaint dataset from CSV file.
    
    Args:
        file_path: Path to the complaints CSV file. If None, uses default path.
        nrows: Number of rows to read (for testing). If None, reads all rows.
        chunk_size: If provided, reads data in chunks. Not recommended for EDA.
    
    Returns:
        DataFrame containing the complaint data.
    
    Raises:
        DataLoadError: If file cannot be loaded or is invalid.
        FileNotFoundError: If the file path does not exist.
    """
    if file_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent
        file_path = project_root / "data" / "raw" / "complaints.csv"
    
    file_path = Path(file_path)
    
    # Validate file exists
    if not file_path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Validate file is readable
    if not os.access(file_path, os.R_OK):
        error_msg = f"File is not readable: {file_path}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)
    
    try:
        logger.info(f"Loading data from {file_path}")
        
        # Read CSV with error handling
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            low_memory=False,
            encoding='utf-8',
            on_bad_lines='skip'  # Skip malformed lines
        )
        
        logger.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Validate required columns exist
        required_columns = ['Product', 'Consumer complaint narrative']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        return df
    
    except pd.errors.EmptyDataError:
        error_msg = f"File is empty: {file_path}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)
    
    except pd.errors.ParserError as e:
        error_msg = f"Error parsing CSV file: {str(e)}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error loading data: {str(e)}"
        logger.error(error_msg)
        raise DataLoadError(error_msg) from e


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate that the loaded dataframe has the expected structure.
    
    Args:
        df: DataFrame to validate.
    
    Returns:
        True if valid, raises DataLoadError otherwise.
    
    Raises:
        DataLoadError: If dataframe is invalid.
    """
    if df is None or df.empty:
        error_msg = "DataFrame is None or empty"
        logger.error(error_msg)
        raise DataLoadError(error_msg)
    
    required_columns = ['Product', 'Consumer complaint narrative']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        error_msg = f"Missing required columns: {missing_columns}"
        logger.error(error_msg)
        raise DataLoadError(error_msg)
    
    logger.info("DataFrame validation passed")
    return True

