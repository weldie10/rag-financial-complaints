"""
Main preprocessing pipeline script.

This script orchestrates the complete EDA and preprocessing workflow.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_cfpb_complaints, validate_dataframe, DataLoadError
from src.data_preprocessor import (
    preprocess_complaints,
    save_processed_data,
    PreprocessingError
)
from src.eda import EDAAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main preprocessing pipeline."""
    try:
        # Step 1: Load data
        logger.info("=" * 60)
        logger.info("Starting CFPB Complaint Data Preprocessing Pipeline")
        logger.info("=" * 60)
        
        df = load_cfpb_complaints()
        validate_dataframe(df)
        
        # Step 2: Perform EDA on raw data
        logger.info("\n" + "=" * 60)
        logger.info("EXPLORATORY DATA ANALYSIS - RAW DATA")
        logger.info("=" * 60)
        
        eda = EDAAnalysis(df)
        
        # Generate and print summary
        summary = eda.generate_summary_report()
        print(summary)
        logger.info("\nEDA Summary:\n" + summary)
        
        # Create visualizations
        project_root = Path(__file__).parent.parent
        plots_dir = project_root / "data" / "processed" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        eda.plot_product_distribution(
            save_path=str(plots_dir / "product_distribution_raw.png")
        )
        eda.plot_narrative_length_distribution(
            save_path=str(plots_dir / "narrative_length_distribution_raw.png")
        )
        
        # Step 3: Preprocess data
        logger.info("\n" + "=" * 60)
        logger.info("DATA PREPROCESSING")
        logger.info("=" * 60)
        
        df_processed = preprocess_complaints(
            df,
            products=None,  # Uses default products
            lowercase=True,
            remove_special_chars=True,
            remove_boilerplate=True,
            normalize_whitespace=True
        )
        
        # Step 4: Perform EDA on processed data
        logger.info("\n" + "=" * 60)
        logger.info("EXPLORATORY DATA ANALYSIS - PROCESSED DATA")
        logger.info("=" * 60)
        
        eda_processed = EDAAnalysis(df_processed)
        summary_processed = eda_processed.generate_summary_report()
        print("\n" + summary_processed)
        logger.info("\nEDA Summary (Processed):\n" + summary_processed)
        
        # Create visualizations for processed data
        eda_processed.plot_product_distribution(
            save_path=str(plots_dir / "product_distribution_processed.png")
        )
        eda_processed.plot_narrative_length_distribution(
            save_path=str(plots_dir / "narrative_length_distribution_processed.png")
        )
        
        # Step 5: Save processed data
        logger.info("\n" + "=" * 60)
        logger.info("SAVING PROCESSED DATA")
        logger.info("=" * 60)
        
        output_path = save_processed_data(df_processed, filename="filtered_complaints.csv")
        
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Processed data saved to: {output_path}")
        logger.info(f"Final dataset size: {len(df_processed):,} rows")
        
        return df_processed
    
    except DataLoadError as e:
        logger.error(f"Data loading error: {e}")
        sys.exit(1)
    
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {e}")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

