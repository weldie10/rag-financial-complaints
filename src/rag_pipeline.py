"""
Main RAG pipeline module.

This module orchestrates the complete RAG pipeline:
1. Retrieval: Embed question and search vector store
2. Prompt Engineering: Format context and question into prompt
3. Generation: Generate response using LLM
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.retriever import RAGRetriever, RetrieverError
from src.prompt_template import PromptTemplate
from src.generator import RAGGenerator, SimpleGenerator, GeneratorError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipelineError(Exception):
    """Custom exception for RAG pipeline errors."""
    pass


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""
    
    def __init__(
        self,
        vector_store_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type: str = "faiss",
        generator_model: Optional[str] = None,
        use_simple_generator: bool = False,
        top_k: int = 5,
        prompt_template: Optional[PromptTemplate] = None,
        **generator_kwargs
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store_path: Path to the vector store directory.
            embedding_model: Embedding model name (must match indexing model).
            vector_store_type: Type of vector store ('faiss' or 'chromadb').
            generator_model: LLM model name. If None, uses default.
            use_simple_generator: If True, uses SimpleGenerator (faster, smaller models).
            top_k: Number of chunks to retrieve (default: 5).
            prompt_template: Custom prompt template. If None, uses default.
            **generator_kwargs: Additional arguments for generator initialization.
        """
        self.top_k = top_k
        
        try:
            # Initialize retriever
            logger.info("Initializing retriever...")
            self.retriever = RAGRetriever(
                vector_store_path=vector_store_path,
                embedding_model=embedding_model,
                vector_store_type=vector_store_type
            )
            
            # Initialize prompt template
            if prompt_template is None:
                self.prompt_template = PromptTemplate.create_analyst_template()
            else:
                self.prompt_template = prompt_template
            
            # Initialize generator
            logger.info("Initializing generator...")
            if use_simple_generator:
                self.generator = SimpleGenerator(
                    model_name=generator_model or "gpt2",
                    **generator_kwargs
                )
            else:
                self.generator = RAGGenerator(
                    model_name=generator_model or "mistralai/Mistral-7B-Instruct-v0.1",
                    **generator_kwargs
                )
            
            logger.info("RAG pipeline initialized successfully")
            
        except RetrieverError as e:
            error_msg = f"Error initializing retriever: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg) from e
        except GeneratorError as e:
            error_msg = f"Error initializing generator: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error initializing pipeline: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg) from e
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> Dict[str, any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's question.
            top_k: Number of chunks to retrieve (overrides default).
            return_sources: Whether to return retrieved sources.
        
        Returns:
            Dictionary containing:
            - 'answer': Generated answer
            - 'sources': List of retrieved chunks (if return_sources=True)
            - 'similarities': List of similarity scores (if return_sources=True)
        
        Raises:
            RAGPipelineError: If query processing fails.
        """
        if not question or not question.strip():
            raise RAGPipelineError("Question cannot be empty")
        
        k = top_k if top_k is not None else self.top_k
        
        try:
            # Step 1: Retrieve relevant chunks
            logger.info(f"Retrieving top-{k} chunks for question...")
            chunks, similarities = self.retriever.retrieve(question, k=k)
            
            if not chunks:
                logger.warning("No chunks retrieved for question")
                return {
                    'answer': "I don't have enough information in the provided context to answer this question.",
                    'sources': [],
                    'similarities': []
                }
            
            # Step 2: Format context
            logger.info("Formatting context...")
            context = self.retriever.format_context(chunks, max_chunks=k)
            
            # Step 3: Create prompt
            logger.info("Creating prompt...")
            prompt = self.prompt_template.format(context=context, question=question)
            
            # Step 4: Generate response
            logger.info("Generating response...")
            answer = self.generator.generate(prompt)
            
            # Prepare result
            result = {
                'answer': answer,
            }
            
            if return_sources:
                result['sources'] = chunks
                result['similarities'] = similarities
            
            logger.info("Query processed successfully")
            return result
            
        except RetrieverError as e:
            error_msg = f"Retrieval error: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg) from e
        except GeneratorError as e:
            error_msg = f"Generation error: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error processing query: {str(e)}"
            logger.error(error_msg)
            raise RAGPipelineError(error_msg) from e
    
    def format_sources(self, sources: List[Dict], max_display: int = 2) -> str:
        """
        Format sources for display in evaluation or UI.
        
        Args:
            sources: List of source chunk dictionaries.
            max_display: Maximum number of sources to display.
        
        Returns:
            Formatted string of sources.
        """
        if not sources:
            return "No sources available."
        
        formatted = []
        for i, source in enumerate(sources[:max_display]):
            complaint_id = source.get('complaint_id', 'N/A')
            product = source.get('product', 'N/A')
            issue = source.get('issue', 'N/A')
            chunk_text_preview = source.get('chunk_text', '')[:200]
            
            formatted.append(
                f"Source {i+1}:\n"
                f"  Complaint ID: {complaint_id}\n"
                f"  Product: {product}\n"
                f"  Issue: {issue}\n"
                f"  Text Preview: {chunk_text_preview}...\n"
            )
        
        if len(sources) > max_display:
            formatted.append(f"\n... and {len(sources) - max_display} more sources")
        
        return "\n".join(formatted)


def create_pipeline(
    vector_store_path: str = "vector_store",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    generator_model: Optional[str] = None,
    use_simple_generator: bool = True,  # Default to simple for easier setup
    **kwargs
) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline with default settings.
    
    Args:
        vector_store_path: Path to vector store.
        embedding_model: Embedding model name.
        generator_model: LLM model name.
        use_simple_generator: Whether to use simple generator.
        **kwargs: Additional pipeline arguments.
    
    Returns:
        Initialized RAGPipeline instance.
    """
    return RAGPipeline(
        vector_store_path=vector_store_path,
        embedding_model=embedding_model,
        generator_model=generator_model,
        use_simple_generator=use_simple_generator,
        **kwargs
    )

