"""
Prompt template module for RAG system.

This module provides prompt templates for guiding the LLM to generate
accurate and helpful responses based on retrieved context.
"""

from typing import Dict, Optional


class PromptTemplate:
    """Template for RAG prompts."""
    
    DEFAULT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:"""
    
    def __init__(self, template: Optional[str] = None):
        """
        Initialize prompt template.
        
        Args:
            template: Custom prompt template. If None, uses default template.
                Template should contain {context} and {question} placeholders.
        """
        self.template = template or self.DEFAULT_TEMPLATE
    
    def format(
        self,
        context: str,
        question: str,
        **kwargs
    ) -> str:
        """
        Format the prompt with context and question.
        
        Args:
            context: Retrieved context chunks formatted as string.
            question: User's question.
            **kwargs: Additional template variables.
        
        Returns:
            Formatted prompt string.
        """
        return self.template.format(
            context=context,
            question=question,
            **kwargs
        )
    
    @classmethod
    def create_analyst_template(cls) -> 'PromptTemplate':
        """
        Create a template optimized for financial analyst role.
        
        Returns:
            PromptTemplate instance with analyst-focused template.
        """
        template = """You are an expert financial analyst assistant for CrediTrust Financial. Your role is to help teams understand customer complaints by analyzing retrieved complaint data.

Instructions:
1. Base your answer ONLY on the provided context from customer complaints
2. If the context contains relevant information, synthesize it into a clear, actionable answer
3. If the context doesn't contain enough information to answer the question, explicitly state: "I don't have enough information in the provided context to answer this question."
4. Be specific and cite relevant details from the complaints when possible
5. Focus on actionable insights that can help improve customer experience

Context from customer complaints:
{context}

Question: {question}

Answer:"""
        
        return cls(template)
    
    @classmethod
    def create_summary_template(cls) -> 'PromptTemplate':
        """
        Create a template optimized for summarization tasks.
        
        Returns:
            PromptTemplate instance with summary-focused template.
        """
        template = """You are a financial analyst assistant for CrediTrust. Summarize the key information from the following customer complaint excerpts related to the question.

Context: {context}

Question: {question}

Provide a concise summary based on the context:"""
        
        return cls(template)

