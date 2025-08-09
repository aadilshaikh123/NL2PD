"""
LLM management module for scalable data agent application.

This module provides an abstraction layer for LLM management that supports
easy switching between different models (Groq, OpenAI, local models, etc.).
Includes retry logic, rate limiting, and factory patterns for scalability.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
import random


@dataclass
class LLMConfig:
    """Configuration class for LLM settings."""
    api_key: str
    model_name: str = "llama3-8b-8192"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    timeout: int = 30
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class RateLimitError(LLMError):
    """Exception raised when rate limits are exceeded."""
    pass


class APIError(LLMError):
    """Exception raised for API-related errors."""
    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the LLM provider."""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        pass
    
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = min(
            self.config.base_delay * (2 ** attempt) + random.uniform(0, 1),
            self.config.max_delay
        )
        return delay


class GroqProvider(LLMProvider):
    """Groq LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None
        
    def initialize(self) -> None:
        """Initialize Groq client."""
        try:
            from langchain_groq import ChatGroq
            
            if not self.config.api_key or self.config.api_key == "your-groq-api-key-goes-here":
                raise LLMError("Groq API key not provided or still using placeholder")
            
            self.client = ChatGroq(
                groq_api_key=self.config.api_key,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            self.logger.info(f"Groq provider initialized with model: {self.config.model_name}")
            
        except ImportError as e:
            raise LLMError(f"Groq dependencies not installed: {e}")
        except Exception as e:
            raise LLMError(f"Failed to initialize Groq provider: {e}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using Groq with retry logic.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not self.client:
            raise LLMError("Groq client not initialized")
        
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Groq generation attempt {attempt + 1}")
                
                # Prepare messages
                messages = [{"role": "user", "content": prompt}]
                
                # Generate response
                response = self.client.invoke(messages)
                
                if hasattr(response, 'content'):
                    result = response.content
                elif hasattr(response, 'text'):
                    result = response.text
                else:
                    result = str(response)
                
                self.logger.info("Groq response generated successfully")
                return result
                
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Handle rate limiting
                if any(term in error_msg for term in ["rate limit", "quota", "429"]):
                    if attempt < self.config.max_retries - 1:
                        delay = self._exponential_backoff(attempt)
                        self.logger.warning(
                            f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1})"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise RateLimitError("Rate limit exceeded after max retries")
                
                # Handle other API errors
                elif any(term in error_msg for term in ["api", "connection", "timeout"]):
                    if attempt < self.config.max_retries - 1:
                        delay = self._exponential_backoff(attempt)
                        self.logger.warning(
                            f"API error, retrying in {delay:.2f}s (attempt {attempt + 1}): {e}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise APIError(f"API error after max retries: {e}")
                
                # Handle other errors
                else:
                    self.logger.error(f"Unexpected error in Groq generation: {e}")
                    raise LLMError(f"Generation failed: {e}")
        
        # If we get here, all retries failed
        raise LLMError(f"Failed after {self.config.max_retries} attempts: {last_exception}")
    
    def is_available(self) -> bool:
        """Check if Groq is available."""
        try:
            return (
                self.client is not None and 
                self.config.api_key and 
                self.config.api_key != "your-groq-api-key-goes-here"
            )
        except Exception:
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation (placeholder for future use)."""
    
    def initialize(self) -> None:
        """Initialize OpenAI client."""
        # Placeholder for OpenAI implementation
        raise NotImplementedError("OpenAI provider not yet implemented")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI."""
        raise NotImplementedError("OpenAI provider not yet implemented")
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return False


class LocalProvider(LLMProvider):
    """Local LLM provider implementation (placeholder for future use)."""
    
    def initialize(self) -> None:
        """Initialize local LLM."""
        # Placeholder for local model implementation
        raise NotImplementedError("Local provider not yet implemented")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using local model."""
        raise NotImplementedError("Local provider not yet implemented")
    
    def is_available(self) -> bool:
        """Check if local model is available."""
        return False


class LLMManager:
    """
    Central LLM management system with provider switching capabilities.
    
    This class provides a unified interface for different LLM providers
    and supports easy switching between them for scalability.
    """
    
    def __init__(self):
        self.providers = {}
        self.active_provider = None
        self.logger = logging.getLogger(__name__)
        
    def register_provider(self, name: str, provider: LLMProvider) -> None:
        """
        Register an LLM provider.
        
        Args:
            name: Provider name
            provider: LLMProvider instance
        """
        self.providers[name] = provider
        self.logger.info(f"Registered LLM provider: {name}")
    
    def set_active_provider(self, name: str) -> None:
        """
        Set the active LLM provider.
        
        Args:
            name: Provider name to activate
        """
        if name not in self.providers:
            raise LLMError(f"Provider '{name}' not registered")
        
        provider = self.providers[name]
        
        if not provider.is_available():
            raise LLMError(f"Provider '{name}' is not available")
        
        self.active_provider = provider
        self.logger.info(f"Active LLM provider set to: {name}")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response using the active provider.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        if not self.active_provider:
            raise LLMError("No active LLM provider set")
        
        return self.active_provider.generate_response(prompt, **kwargs)
    
    def get_available_providers(self) -> Dict[str, bool]:
        """
        Get list of available providers.
        
        Returns:
            Dictionary mapping provider names to availability status
        """
        return {
            name: provider.is_available()
            for name, provider in self.providers.items()
        }
    
    def get_active_provider_name(self) -> Optional[str]:
        """Get the name of the active provider."""
        for name, provider in self.providers.items():
            if provider == self.active_provider:
                return name
        return None


class LLMFactory:
    """Factory class for creating LLM providers and managers."""
    
    @staticmethod
    def create_groq_provider(
        api_key: str,
        model_name: str = "llama3-8b-8192",
        **config_kwargs
    ) -> GroqProvider:
        """
        Create a Groq provider instance.
        
        Args:
            api_key: Groq API key
            model_name: Model name to use
            **config_kwargs: Additional configuration options
            
        Returns:
            Configured GroqProvider instance
        """
        config = LLMConfig(
            api_key=api_key,
            model_name=model_name,
            **config_kwargs
        )
        provider = GroqProvider(config)
        provider.initialize()
        return provider
    
    @staticmethod
    def create_manager_with_groq(
        api_key: str,
        model_name: str = "llama3-8b-8192",
        **config_kwargs
    ) -> LLMManager:
        """
        Create an LLM manager with Groq provider configured.
        
        Args:
            api_key: Groq API key
            model_name: Model name to use
            **config_kwargs: Additional configuration options
            
        Returns:
            Configured LLMManager instance
        """
        manager = LLMManager()
        
        try:
            groq_provider = LLMFactory.create_groq_provider(
                api_key=api_key,
                model_name=model_name,
                **config_kwargs
            )
            manager.register_provider("groq", groq_provider)
            manager.set_active_provider("groq")
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to setup Groq provider: {e}")
            raise LLMError(f"Failed to create manager with Groq: {e}")
        
        return manager
    
    @staticmethod
    def create_manager_from_env() -> LLMManager:
        """
        Create LLM manager from environment variables.
        
        Returns:
            Configured LLMManager instance
        """
        api_key = os.getenv("GROQ_API_KEY")
        
        if not api_key:
            raise LLMError("GROQ_API_KEY environment variable not set")
        
        model_name = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        temperature = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
        max_retries = int(os.getenv("GROQ_MAX_RETRIES", "5"))
        
        return LLMFactory.create_manager_with_groq(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_retries=max_retries
        )


# Convenience functions for easy usage
def create_llm_manager(api_key: str, **kwargs) -> LLMManager:
    """
    Convenience function to create LLM manager.
    
    Args:
        api_key: LLM API key
        **kwargs: Additional configuration options
        
    Returns:
        Configured LLMManager instance
    """
    return LLMFactory.create_manager_with_groq(api_key=api_key, **kwargs)


def create_llm_manager_from_env() -> LLMManager:
    """
    Convenience function to create LLM manager from environment.
    
    Returns:
        Configured LLMManager instance
    """
    return LLMFactory.create_manager_from_env()
