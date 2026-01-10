"""
Generator module for RAG system.

This module implements the generation component that uses LLMs to generate
responses based on prompts containing retrieved context.
"""

import logging
from typing import Optional, Dict, Any

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from langchain.llms import HuggingFacePipeline
    from langchain import PromptTemplate as LangChainPromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeneratorError(Exception):
    """Custom exception for generator errors."""
    pass


class RAGGenerator:
    """Generator for RAG system that uses LLMs to generate responses."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        use_langchain: bool = False,
        device: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize RAG generator.
        
        Args:
            model_name: Name of the LLM model to use.
                Options: "mistralai/Mistral-7B-Instruct-v0.1", 
                         "meta-llama/Llama-2-7b-chat-hf",
                         "microsoft/DialoGPT-medium" (smaller, faster),
                         or any Hugging Face model.
            use_langchain: Whether to use LangChain wrapper (if available).
            device: Device to use ('cpu', 'cuda', or None for auto).
            temperature: Sampling temperature (0.0-1.0, higher = more creative).
            max_new_tokens: Maximum number of tokens to generate.
            load_in_8bit: Whether to load model in 8-bit mode (requires bitsandbytes).
            load_in_4bit: Whether to load model in 4-bit mode (requires bitsandbytes).
        
        Raises:
            GeneratorError: If initialization fails.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise GeneratorError(
                "transformers is not installed. "
                "Please install it: pip install transformers accelerate"
            )
        
        self.model_name = model_name
        self.use_langchain = use_langchain and LANGCHAIN_AVAILABLE
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        try:
            # Determine device
            if device is None:
                try:
                    import torch
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                except ImportError:
                    device = 'cpu'
            self.device = device
            
            logger.info(f"Initializing generator with model: {model_name}")
            logger.info(f"Using device: {device}")
            logger.info(f"Using LangChain: {self.use_langchain}")
            
            if self.use_langchain and LANGCHAIN_AVAILABLE:
                self._init_langchain_model(load_in_8bit, load_in_4bit)
            else:
                self._init_transformers_pipeline(load_in_8bit, load_in_4bit)
            
            logger.info("Generator initialized successfully")
            
        except Exception as e:
            error_msg = f"Error initializing generator: {str(e)}"
            logger.error(error_msg)
            raise GeneratorError(error_msg) from e
    
    def _init_transformers_pipeline(
        self,
        load_in_8bit: bool,
        load_in_4bit: bool
    ):
        """Initialize transformers pipeline."""
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            if not TRANSFORMERS_AVAILABLE:
                logger.warning("bitsandbytes not available, skipping quantization")
            else:
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=load_in_8bit,
                        load_in_4bit=load_in_4bit
                    )
                except:
                    logger.warning("Could not configure quantization, using full precision")
        
        # Use text-generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            device=0 if self.device == 'cuda' else -1,
            model_kwargs={
                "quantization_config": quantization_config,
                "torch_dtype": "auto",
                "trust_remote_code": True
            } if quantization_config else {
                "torch_dtype": "auto",
                "trust_remote_code": True
            },
            return_full_text=False
        )
    
    def _init_langchain_model(
        self,
        load_in_8bit: bool,
        load_in_4bit: bool
    ):
        """Initialize LangChain model."""
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit
                )
            except:
                logger.warning("Could not configure quantization")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == 'cuda' else None,
            torch_dtype="auto",
            trust_remote_code=True
        )
        
        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            return_full_text=False
        )
        
        self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Formatted prompt string.
            **kwargs: Additional generation parameters.
        
        Returns:
            Generated response text.
        
        Raises:
            GeneratorError: If generation fails.
        """
        if not prompt or not prompt.strip():
            raise GeneratorError("Prompt cannot be empty")
        
        try:
            temperature = kwargs.get('temperature', self.temperature)
            max_new_tokens = kwargs.get('max_new_tokens', self.max_new_tokens)
            
            if self.use_langchain and LANGCHAIN_AVAILABLE:
                # Use LangChain
                response = self.llm(prompt, temperature=temperature)
                return response.strip()
            else:
                # Use transformers pipeline directly
                outputs = self.pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    num_return_sequences=1,
                    pad_token_id=self.pipeline.tokenizer.eos_token_id
                )
                
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].get('generated_text', '')
                    return generated_text.strip()
                else:
                    return ""
                    
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            logger.error(error_msg)
            raise GeneratorError(error_msg) from e


class SimpleGenerator:
    """
    Simple generator using smaller, faster models for testing.
    Uses models that are easier to run on CPU or limited resources.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",  # Small, fast model for testing
        temperature: float = 0.7,
        max_length: int = 200
    ):
        """
        Initialize simple generator.
        
        Args:
            model_name: Name of a small model (e.g., "gpt2", "distilgpt2").
            temperature: Sampling temperature.
            max_length: Maximum generation length.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise GeneratorError("transformers is not installed")
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_length = max_length
        
        try:
            logger.info(f"Loading simple generator model: {model_name}")
            self.pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=-1,  # CPU
                model_kwargs={"pad_token_id": None}  # Will set after loading
            )
            # Set pad token if not set
            if self.pipeline.tokenizer.pad_token is None:
                self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
            logger.info("Simple generator initialized")
        except Exception as e:
            raise GeneratorError(f"Error initializing simple generator: {str(e)}") from e
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response."""
        try:
            max_length = kwargs.get('max_length', self.max_length)
            # Calculate total length (prompt + new tokens)
            prompt_length = len(self.pipeline.tokenizer.encode(prompt))
            max_new_tokens = max_length - prompt_length
            
            if max_new_tokens <= 0:
                max_new_tokens = 50  # Minimum generation length
            
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
                temperature=kwargs.get('temperature', self.temperature),
                num_return_sequences=1,
                pad_token_id=self.pipeline.tokenizer.pad_token_id or self.pipeline.tokenizer.eos_token_id,
                do_sample=self.temperature > 0.0
            )
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].get('generated_text', '')
                # Remove the prompt from the beginning
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                return generated_text
            return ""
        except Exception as e:
            raise GeneratorError(f"Error during generation: {str(e)}") from e

