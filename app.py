"""
Main application file for the RAG Financial Complaints system.
Interactive Gradio web interface for querying the RAG system.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline, create_pipeline, RAGPipelineError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline instance (initialized on startup)
pipeline: Optional[RAGPipeline] = None


def initialize_pipeline(
    vector_store_path: str = "vector_store",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    generator_model: Optional[str] = None,
    use_simple_generator: bool = True,
    **kwargs
) -> RAGPipeline:
    """
    Initialize the RAG pipeline.
    
    Args:
        vector_store_path: Path to vector store directory.
        embedding_model: Embedding model name.
        generator_model: LLM model name (optional).
        use_simple_generator: Whether to use simple generator.
        **kwargs: Additional pipeline arguments.
    
    Returns:
        Initialized RAGPipeline instance.
    """
    try:
        logger.info("Initializing RAG pipeline...")
        pipeline = create_pipeline(
            vector_store_path=vector_store_path,
            embedding_model=embedding_model,
            generator_model=generator_model,
            use_simple_generator=use_simple_generator,
            **kwargs
        )
        logger.info("RAG pipeline initialized successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Error initializing pipeline: {str(e)}")
        raise


def format_sources_for_display(sources: list, similarities: list, max_display: int = 5) -> str:
    """
    Format source chunks for display in the UI.
    
    Args:
        sources: List of source chunk dictionaries.
        similarities: List of similarity scores.
        max_display: Maximum number of sources to display.
    
    Returns:
        Formatted HTML string for display.
    """
    if not sources:
        return "**No sources available.**"
    
    formatted_sources = []
    for i, (source, similarity) in enumerate(zip(sources[:max_display], similarities[:max_display])):
        complaint_id = source.get('complaint_id', 'N/A')
        product = source.get('product', 'N/A')
        issue = source.get('issue', 'N/A')
        chunk_text = source.get('chunk_text', '')
        
        # Truncate long text for display
        text_preview = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
        
        formatted_sources.append(
            f"### Source {i+1} (Similarity: {similarity:.3f})\n\n"
            f"**Complaint ID:** {complaint_id}\n\n"
            f"**Product:** {product}\n\n"
            f"**Issue:** {issue}\n\n"
            f"**Text:**\n{text_preview}\n\n"
            f"---\n"
        )
    
    if len(sources) > max_display:
        formatted_sources.append(f"\n*... and {len(sources) - max_display} more sources*")
    
    return "\n".join(formatted_sources)


def query_rag_streaming(question: str, history: list):
    """
    Process a query through the RAG pipeline with streaming support.
    
    Args:
        question: User's question.
        history: Chat history (for future use).
    
    Yields:
        Tuple of (partial_answer, sources_display, updated_history).
    """
    global pipeline
    
    if not question or not question.strip():
        yield "Please enter a question.", "No sources available.", history
        return
    
    if pipeline is None:
        yield (
            "Error: RAG pipeline not initialized. Please check the logs.",
            "No sources available.",
            history
        )
        return
    
    try:
        logger.info(f"Processing query: {question[:100]}...")
        
        # Query the pipeline
        result = pipeline.query(question, return_sources=True)
        
        answer = result.get('answer', 'No answer generated.')
        sources = result.get('sources', [])
        similarities = result.get('similarities', [])
        
        # Format sources for display
        sources_display = format_sources_for_display(sources, similarities)
        
        # Stream the answer token by token (simulate streaming)
        # Split by words and yield progressively
        words = answer.split()
        partial_answer = ""
        
        # First yield: show sources immediately (user can see what was retrieved)
        if words:
            yield "Retrieving and generating answer...", sources_display, history
        
        for i, word in enumerate(words):
            partial_answer += word + " "
            # Yield every few words for smoother streaming
            if i % 3 == 0 or i == len(words) - 1:
                # Update history with final answer only on last iteration
                if i == len(words) - 1:
                    history.append((question, answer))
                    yield partial_answer.strip(), sources_display, history
                else:
                    yield partial_answer.strip(), sources_display, history
        
        logger.info("Query processed successfully")
        
    except RAGPipelineError as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        yield error_msg, "No sources available.", history
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        yield error_msg, "No sources available.", history


def query_rag(question: str, history: list) -> Tuple[str, str, list]:
    """
    Process a query through the RAG pipeline (non-streaming version).
    
    Args:
        question: User's question.
        history: Chat history (for future use).
    
    Returns:
        Tuple of (answer, sources_display, updated_history).
    """
    global pipeline
    
    if not question or not question.strip():
        return "Please enter a question.", "No sources available.", history
    
    if pipeline is None:
        return (
            "Error: RAG pipeline not initialized. Please check the logs.",
            "No sources available.",
            history
        )
    
    try:
        logger.info(f"Processing query: {question[:100]}...")
        
        # Query the pipeline
        result = pipeline.query(question, return_sources=True)
        
        answer = result.get('answer', 'No answer generated.')
        sources = result.get('sources', [])
        similarities = result.get('similarities', [])
        
        # Format sources for display
        sources_display = format_sources_for_display(sources, similarities)
        
        # Update history (for future conversation support)
        history.append((question, answer))
        
        logger.info("Query processed successfully")
        return answer, sources_display, history
        
    except RAGPipelineError as e:
        error_msg = f"Error processing query: {str(e)}"
        logger.error(error_msg)
        return error_msg, "No sources available.", history
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return error_msg, "No sources available.", history


def clear_conversation() -> Tuple[str, str, str, list]:
    """
    Clear the conversation history and all displays.
    
    Returns:
        Tuple of (empty_question, empty_answer, empty_sources, empty_history).
    """
    default_sources = "**Sources will appear here after you ask a question.**"
    return "", "", default_sources, []


def create_interface() -> gr.Blocks:
    """
    Create the Gradio interface.
    
    Returns:
        Gradio Blocks interface.
    """
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
    }
    """
    
    with gr.Blocks(title="RAG Financial Complaints Assistant", css=custom_css, theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🤖 RAG Financial Complaints Assistant
            
            Ask questions about customer complaints in the financial services dataset.
            The system will retrieve relevant complaint information and generate an answer.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Question input
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What are the most common issues with credit card complaints?",
                    lines=3,
                    interactive=True
                )
                
                # Buttons
                with gr.Row():
                    submit_btn = gr.Button("Ask Question", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", variant="secondary", size="lg")
                
                # Answer display
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=10,
                    interactive=False,
                    show_copy_button=True
                )
            
            with gr.Column(scale=1):
                # Sources display
                sources_output = gr.Markdown(
                    label="Retrieved Sources",
                    value="**Sources will appear here after you ask a question.**"
                )
        
        # Chat history (hidden, for future use)
        chat_history = gr.State(value=[])
        
        # Event handlers with streaming support
        submit_btn.click(
            fn=query_rag_streaming,
            inputs=[question_input, chat_history],
            outputs=[answer_output, sources_output, chat_history],
            show_progress=True
        )
        
        # Allow Enter key to submit (with streaming)
        question_input.submit(
            fn=query_rag_streaming,
            inputs=[question_input, chat_history],
            outputs=[answer_output, sources_output, chat_history],
            show_progress=True
        )
        
        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[question_input, answer_output, sources_output, chat_history]
        )
        
        # Examples
        gr.Examples(
            examples=[
                "What are the most common issues with credit card complaints?",
                "What problems do customers face with personal loans?",
                "How do customers describe issues with savings accounts?",
                "What are the main complaints about money transfer services?",
                "What are customers saying about unauthorized charges?",
            ],
            inputs=question_input
        )
        
        gr.Markdown(
            """
            ---
            ### 📝 Instructions
            1. Type your question in the text box above
            2. Click "Ask Question" or press Enter
            3. Review the generated answer
            4. Check the sources below to see which complaint chunks were used
            5. Use "Clear" to reset and ask a new question
            
            ### ℹ️ About
            This interface uses a Retrieval-Augmented Generation (RAG) system to answer questions
            about financial complaints. The system retrieves relevant complaint excerpts and uses
            them to generate accurate, context-aware answers.
            """
        )
    
    return interface


def main():
    """Main application entry point."""
    global pipeline
    
    # Configuration
    vector_store_path = "vector_store"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Use simple generator by default for easier setup
    # Set to False and provide a model name to use full models
    use_simple_generator = True
    generator_model = "gpt2"  # Small model for testing
    
    try:
        # Initialize pipeline
        logger.info("Starting RAG Financial Complaints Assistant...")
        pipeline = initialize_pipeline(
            vector_store_path=vector_store_path,
            embedding_model=embedding_model,
            generator_model=generator_model,
            use_simple_generator=use_simple_generator
        )
        
        # Create and launch interface
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,        # Default Gradio port
            share=False,             # Set to True to create a public link
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
