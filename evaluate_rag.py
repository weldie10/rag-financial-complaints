"""
Main evaluation script for RAG pipeline.

This script evaluates the RAG system with representative questions
and generates an evaluation report.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline, create_pipeline
from src.evaluation import run_evaluation, DEFAULT_EVALUATION_QUESTIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main evaluation function."""
    
    # Configuration
    vector_store_path = "vector_store"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Use simple generator by default for easier setup
    # Change to False and provide a model name to use full models
    use_simple_generator = True
    generator_model = "gpt2"  # Small model for testing
    
    # Evaluation questions
    evaluation_questions = [
        "What are the most common issues with credit card complaints?",
        "What problems do customers face with personal loans?",
        "How do customers describe issues with savings accounts?",
        "What are the main complaints about money transfer services?",
        "Which companies receive the most complaints about credit cards?",
        "What are customers saying about unauthorized charges?",
        "What issues are related to account management for savings accounts?",
        "How do customers describe problems with loan servicing?",
        "What are common complaints about transaction processing?",
        "What do customers say about customer service quality?"
    ]
    
    # Quality scores (1-5) - to be filled after manual review
    # These are placeholders; actual scores should be assigned after reviewing answers
    quality_scores = [None] * len(evaluation_questions)  # Will be filled after evaluation
    
    # Comments - to be filled after manual review
    comments = [None] * len(evaluation_questions)  # Will be filled after evaluation
    
    try:
        logger.info("=" * 60)
        logger.info("RAG Pipeline Evaluation")
        logger.info("=" * 60)
        
        # Check if vector store exists
        vector_store_dir = Path(vector_store_path)
        if not vector_store_dir.exists():
            logger.error(f"Vector store directory not found at {vector_store_path}")
            logger.error("Please run the indexing pipeline first:")
            logger.error("  python src/indexing_pipeline.py")
            sys.exit(1)
        
        # Check for required files
        required_files = ["faiss.index", "metadata.pkl", "index_info.pkl"]
        missing_files = [f for f in required_files if not (vector_store_dir / f).exists()]
        if missing_files:
            logger.error(f"Vector store files missing: {missing_files}")
            logger.error("Please run the indexing pipeline first:")
            logger.error("  python src/indexing_pipeline.py")
            sys.exit(1)
        
        # Initialize pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = create_pipeline(
            vector_store_path=vector_store_path,
            embedding_model=embedding_model,
            generator_model=generator_model,
            use_simple_generator=use_simple_generator,
            top_k=5
        )
        
        # Run evaluation
        logger.info("Running evaluation...")
        results, table = run_evaluation(
            pipeline=pipeline,
            questions=evaluation_questions,
            output_path="evaluation_results",
            quality_scores=quality_scores,
            comments=comments
        )
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print("\n" + table)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total questions evaluated: {len(results)}")
        print(f"Questions with sources: {sum(1 for r in results if r.get('sources'))}")
        print(f"Average sources per question: {sum(len(r.get('sources', [])) for r in results) / len(results):.2f}")
        
        logger.info("Evaluation completed successfully")
        logger.info("Results saved to evaluation_results.csv and evaluation_results.md")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

