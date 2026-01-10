"""
Evaluation module for RAG system.

This module provides functionality to evaluate the RAG pipeline
with representative questions and generate evaluation reports.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.rag_pipeline import RAGPipeline, RAGPipelineError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Custom exception for evaluation errors."""
    pass


class RAGEvaluator:
    """Evaluator for RAG pipeline."""
    
    def __init__(self, pipeline: RAGPipeline):
        """
        Initialize evaluator.
        
        Args:
            pipeline: Initialized RAGPipeline instance.
        """
        self.pipeline = pipeline
    
    def evaluate_question(
        self,
        question: str,
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate a single question.
        
        Args:
            question: Question to evaluate.
            top_k: Number of chunks to retrieve.
        
        Returns:
            Dictionary with evaluation results:
            - 'question': The question
            - 'answer': Generated answer
            - 'sources': Retrieved sources
            - 'similarities': Similarity scores
        """
        try:
            result = self.pipeline.query(question, top_k=top_k, return_sources=True)
            return {
                'question': question,
                'answer': result['answer'],
                'sources': result.get('sources', []),
                'similarities': result.get('similarities', [])
            }
        except Exception as e:
            logger.error(f"Error evaluating question '{question}': {str(e)}")
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'sources': [],
                'similarities': []
            }
    
    def evaluate_questions(
        self,
        questions: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Evaluate multiple questions.
        
        Args:
            questions: List of questions to evaluate.
            top_k: Number of chunks to retrieve per question.
        
        Returns:
            List of evaluation result dictionaries.
        """
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: {question[:50]}...")
            result = self.evaluate_question(question, top_k=top_k)
            results.append(result)
        return results
    
    def format_evaluation_table(
        self,
        results: List[Dict],
        quality_scores: Optional[List[int]] = None,
        comments: Optional[List[str]] = None
    ) -> str:
        """
        Format evaluation results as a Markdown table.
        
        Args:
            results: List of evaluation result dictionaries.
            quality_scores: Optional list of quality scores (1-5) for each result.
            comments: Optional list of comments/analysis for each result.
        
        Returns:
            Markdown-formatted table string.
        """
        table_rows = []
        
        # Header
        table_rows.append("| Question | Generated Answer | Retrieved Sources | Quality Score | Comments/Analysis |")
        table_rows.append("|----------|-------------------|-------------------|---------------|-------------------|")
        
        for i, result in enumerate(results):
            question = result['question']
            answer = result['answer']
            
            # Truncate answer if too long
            if len(answer) > 200:
                answer_display = answer[:200] + "..."
            else:
                answer_display = answer
            
            # Format sources
            sources = result.get('sources', [])
            if sources:
                source_display = self._format_source_for_table(sources[0])
                if len(sources) > 1:
                    source_display += f"<br>(+{len(sources)-1} more)"
            else:
                source_display = "No sources"
            
            # Quality score
            score = quality_scores[i] if quality_scores and i < len(quality_scores) else "N/A"
            
            # Comments
            comment = comments[i] if comments and i < len(comments) else ""
            
            # Escape pipe characters in markdown
            question = question.replace("|", "\\|")
            answer_display = answer_display.replace("|", "\\|")
            source_display = source_display.replace("|", "\\|")
            comment = comment.replace("|", "\\|")
            
            table_rows.append(
                f"| {question} | {answer_display} | {source_display} | {score} | {comment} |"
            )
        
        return "\n".join(table_rows)
    
    def _format_source_for_table(self, source: Dict, max_length: int = 150) -> str:
        """Format a single source for table display."""
        complaint_id = source.get('complaint_id', 'N/A')
        product = source.get('product', 'N/A')
        text_preview = source.get('chunk_text', '')[:max_length]
        
        return f"ID: {complaint_id}, Product: {product}<br>Text: {text_preview}..."


# Default evaluation questions
DEFAULT_EVALUATION_QUESTIONS = [
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


def run_evaluation(
    pipeline: RAGPipeline,
    questions: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    quality_scores: Optional[List[int]] = None,
    comments: Optional[List[str]] = None
) -> Tuple[List[Dict], str]:
    """
    Run complete evaluation and generate report.
    
    Args:
        pipeline: Initialized RAGPipeline instance.
        questions: List of questions to evaluate. If None, uses default questions.
        output_path: Optional path to save evaluation results.
        quality_scores: Optional quality scores for each question.
        comments: Optional comments for each question.
    
    Returns:
        Tuple of (results_list, markdown_table_string).
    """
    if questions is None:
        questions = DEFAULT_EVALUATION_QUESTIONS
    
    evaluator = RAGEvaluator(pipeline)
    
    logger.info(f"Starting evaluation with {len(questions)} questions...")
    results = evaluator.evaluate_questions(questions)
    
    logger.info("Formatting evaluation table...")
    table = evaluator.format_evaluation_table(results, quality_scores, comments)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results as CSV
        csv_path = output_path.with_suffix('.csv')
        df_results = pd.DataFrame([
            {
                'question': r['question'],
                'answer': r['answer'],
                'num_sources': len(r.get('sources', [])),
                'avg_similarity': sum(r.get('similarities', [])) / len(r.get('similarities', [])) if r.get('similarities') else 0
            }
            for r in results
        ])
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Saved evaluation results to {csv_path}")
        
        # Save markdown table
        md_path = output_path.with_suffix('.md')
        with open(md_path, 'w') as f:
            f.write("# RAG Pipeline Evaluation Results\n\n")
            f.write(table)
        logger.info(f"Saved evaluation table to {md_path}")
    
    return results, table

