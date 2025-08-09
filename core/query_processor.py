"""
Query processing module for scalable data agent application.

This module provides an abstraction layer for query processing that can be
easily switched from pandas-ai to custom pipelines in the future. Includes
interface definitions and plugin architecture support.
"""

import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
import os
import tempfile
from pathlib import Path


@dataclass
class QueryResult:
    """Standardized query result structure."""
    success: bool
    result: Any
    result_type: str  # 'dataframe', 'text', 'chart', 'error'
    execution_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryProcessorError(Exception):
    """Base exception for query processing errors."""
    pass


class QueryProcessor(ABC):
    """Abstract base class for query processors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the query processor."""
        pass
    
    @abstractmethod
    def process_query(self, query: str, dataframe: pd.DataFrame, **kwargs) -> QueryResult:
        """Process a natural language query against a DataFrame."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the query processor is available."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of processor capabilities."""
        pass


class PandasAIProcessor(QueryProcessor):
    """PandasAI implementation of query processor."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smart_dataframe = None
        self.llm = None
        
    def initialize(self) -> None:
        """Initialize PandasAI with LLM backend."""
        try:
            import pandasai as pai
            from pandasai.llm import LangchainLLM
            
            # Get LLM from config
            llm_manager = self.config.get("llm_manager")
            if not llm_manager:
                raise QueryProcessorError("LLM manager not provided in config")
            
            # Create LangchainLLM wrapper for PandasAI
            self.llm = LangchainLLM(llm_manager.active_provider.client)
            
            self.logger.info("PandasAI processor initialized successfully")
            
        except ImportError as e:
            raise QueryProcessorError(f"PandasAI dependencies not installed: {e}")
        except Exception as e:
            raise QueryProcessorError(f"Failed to initialize PandasAI: {e}")
    
    def _create_smart_dataframe(self, dataframe: pd.DataFrame) -> Any:
        """Create SmartDataframe instance."""
        try:
            import pandasai as pai
            
            # Configure PandasAI
            config = {
                "llm": self.llm,
                "verbose": self.config.get("verbose", False),
                "conversational": self.config.get("conversational", True),
                "enable_cache": self.config.get("enable_cache", True),
                "save_charts": self.config.get("save_charts", False),
                "save_charts_path": self.config.get("charts_path", "charts/")
            }
            
            # Create SmartDataframe
            smart_df = pai.SmartDataframe(dataframe, config=config)
            
            return smart_df
            
        except Exception as e:
            raise QueryProcessorError(f"Failed to create SmartDataframe: {e}")
    
    def process_query(self, query: str, dataframe: pd.DataFrame, **kwargs) -> QueryResult:
        """
        Process query using PandasAI.
        
        Args:
            query: Natural language query
            dataframe: DataFrame to query
            **kwargs: Additional processing options
            
        Returns:
            QueryResult object
        """
        import time
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query with PandasAI: {query[:100]}...")
            
            # Create SmartDataframe
            smart_df = self._create_smart_dataframe(dataframe)
            
            # Process query
            result = smart_df.chat(query)
            
            execution_time = time.time() - start_time
            
            # Determine result type
            result_type = self._determine_result_type(result)
            
            # Create successful result
            query_result = QueryResult(
                success=True,
                result=result,
                result_type=result_type,
                execution_time=execution_time,
                metadata={
                    "processor": "pandasai",
                    "dataframe_shape": dataframe.shape,
                    "query_length": len(query)
                }
            )
            
            self.logger.info(f"Query processed successfully in {execution_time:.2f}s")
            return query_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"Query processing failed: {error_msg}")
            
            return QueryResult(
                success=False,
                result=None,
                result_type="error",
                execution_time=execution_time,
                error_message=error_msg,
                metadata={
                    "processor": "pandasai",
                    "dataframe_shape": dataframe.shape,
                    "query_length": len(query)
                }
            )
    
    def _determine_result_type(self, result: Any) -> str:
        """Determine the type of result returned by PandasAI."""
        if isinstance(result, pd.DataFrame):
            return "dataframe"
        elif isinstance(result, str) and any(ext in result for ext in ['.png', '.jpg', '.jpeg', '.svg']):
            return "chart"
        elif isinstance(result, (int, float, str, bool)):
            return "text"
        else:
            return "unknown"
    
    def is_available(self) -> bool:
        """Check if PandasAI is available."""
        try:
            import pandasai
            return self.llm is not None
        except ImportError:
            return False
    
    def get_capabilities(self) -> List[str]:
        """Get PandasAI capabilities."""
        return [
            "natural_language_queries",
            "data_analysis",
            "chart_generation",
            "statistical_analysis",
            "data_filtering",
            "aggregations",
            "conversational_interface"
        ]


class CustomProcessor(QueryProcessor):
    """Custom query processor implementation (placeholder for future use)."""
    
    def initialize(self) -> None:
        """Initialize custom processor."""
        raise NotImplementedError("Custom processor not yet implemented")
    
    def process_query(self, query: str, dataframe: pd.DataFrame, **kwargs) -> QueryResult:
        """Process query using custom logic."""
        raise NotImplementedError("Custom processor not yet implemented")
    
    def is_available(self) -> bool:
        """Check if custom processor is available."""
        return False
    
    def get_capabilities(self) -> List[str]:
        """Get custom processor capabilities."""
        return []


class QueryProcessorManager:
    """
    Central query processor management system.
    
    This class provides a unified interface for different query processors
    and supports easy switching between them for scalability.
    """
    
    def __init__(self):
        self.processors = {}
        self.active_processor = None
        self.logger = logging.getLogger(__name__)
        
    def register_processor(self, name: str, processor: QueryProcessor) -> None:
        """
        Register a query processor.
        
        Args:
            name: Processor name
            processor: QueryProcessor instance
        """
        self.processors[name] = processor
        self.logger.info(f"Registered query processor: {name}")
    
    def set_active_processor(self, name: str) -> None:
        """
        Set the active query processor.
        
        Args:
            name: Processor name to activate
        """
        if name not in self.processors:
            raise QueryProcessorError(f"Processor '{name}' not registered")
        
        processor = self.processors[name]
        
        if not processor.is_available():
            raise QueryProcessorError(f"Processor '{name}' is not available")
        
        self.active_processor = processor
        self.logger.info(f"Active query processor set to: {name}")
    
    def process_query(self, query: str, dataframe: pd.DataFrame, **kwargs) -> QueryResult:
        """
        Process query using the active processor.
        
        Args:
            query: Natural language query
            dataframe: DataFrame to query
            **kwargs: Additional parameters
            
        Returns:
            QueryResult object
        """
        if not self.active_processor:
            raise QueryProcessorError("No active query processor set")
        
        return self.active_processor.process_query(query, dataframe, **kwargs)
    
    def get_available_processors(self) -> Dict[str, bool]:
        """
        Get list of available processors.
        
        Returns:
            Dictionary mapping processor names to availability status
        """
        return {
            name: processor.is_available()
            for name, processor in self.processors.items()
        }
    
    def get_active_processor_name(self) -> Optional[str]:
        """Get the name of the active processor."""
        for name, processor in self.processors.items():
            if processor == self.active_processor:
                return name
        return None
    
    def get_processor_capabilities(self, name: Optional[str] = None) -> List[str]:
        """
        Get capabilities of a processor.
        
        Args:
            name: Processor name (uses active processor if None)
            
        Returns:
            List of capabilities
        """
        if name:
            if name not in self.processors:
                raise QueryProcessorError(f"Processor '{name}' not registered")
            return self.processors[name].get_capabilities()
        elif self.active_processor:
            return self.active_processor.get_capabilities()
        else:
            return []


class QueryProcessorFactory:
    """Factory class for creating query processors and managers."""
    
    @staticmethod
    def create_pandasai_processor(llm_manager: Any, **config_kwargs) -> PandasAIProcessor:
        """
        Create a PandasAI processor instance.
        
        Args:
            llm_manager: LLM manager instance
            **config_kwargs: Additional configuration options
            
        Returns:
            Configured PandasAIProcessor instance
        """
        config = {
            "llm_manager": llm_manager,
            "verbose": config_kwargs.get("verbose", False),
            "conversational": config_kwargs.get("conversational", True),
            "enable_cache": config_kwargs.get("enable_cache", True),
            "save_charts": config_kwargs.get("save_charts", False),
            **config_kwargs
        }
        
        processor = PandasAIProcessor(config)
        processor.initialize()
        return processor
    
    @staticmethod
    def create_manager_with_pandasai(llm_manager: Any, **config_kwargs) -> QueryProcessorManager:
        """
        Create a query processor manager with PandasAI configured.
        
        Args:
            llm_manager: LLM manager instance
            **config_kwargs: Additional configuration options
            
        Returns:
            Configured QueryProcessorManager instance
        """
        manager = QueryProcessorManager()
        
        try:
            pandasai_processor = QueryProcessorFactory.create_pandasai_processor(
                llm_manager=llm_manager,
                **config_kwargs
            )
            manager.register_processor("pandasai", pandasai_processor)
            manager.set_active_processor("pandasai")
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to setup PandasAI processor: {e}")
            raise QueryProcessorError(f"Failed to create manager with PandasAI: {e}")
        
        return manager


# Convenience functions for easy usage
def create_query_processor_manager(llm_manager: Any, **kwargs) -> QueryProcessorManager:
    """
    Convenience function to create query processor manager.
    
    Args:
        llm_manager: LLM manager instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured QueryProcessorManager instance
    """
    return QueryProcessorFactory.create_manager_with_pandasai(
        llm_manager=llm_manager,
        **kwargs
    )
