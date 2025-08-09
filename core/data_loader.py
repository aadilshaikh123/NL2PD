"""
Data loading and processing module for scalable data agent application.

This module provides a comprehensive data loading system with caching,
validation, and preprocessing capabilities. Designed for easy migration
to backend API architecture.
"""

import pandas as pd
import logging
from typing import Union, Dict, Any, Optional
import os
from pathlib import Path
import hashlib


class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass


class DataLoader:
    """
    Scalable data loading system with caching and validation.
    
    This class handles CSV and Excel file loading with comprehensive
    error handling, validation, and preprocessing capabilities.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize DataLoader with caching capabilities.
        
        Args:
            cache_dir: Directory for caching processed data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def _generate_cache_key(self, file_path: str, **kwargs) -> str:
        """Generate a unique cache key for the file and parameters."""
        file_stat = os.stat(file_path)
        content = f"{file_path}_{file_stat.st_mtime}_{file_stat.st_size}_{str(kwargs)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded DataFrame and return metadata.
        
        Args:
            df: Loaded DataFrame
            
        Returns:
            Dictionary containing validation results and metadata
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
        }
        
        # Check for empty DataFrame
        if df.empty:
            validation_results["errors"].append("DataFrame is empty")
            validation_results["is_valid"] = False
            
        # Check for excessive null values
        null_percentage = (df.isnull().sum() / len(df)) * 100
        high_null_cols = null_percentage[null_percentage > 50].index.tolist()
        if high_null_cols:
            validation_results["warnings"].append(
                f"Columns with >50% null values: {high_null_cols}"
            )
            
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["warnings"].append(
                f"Found {duplicate_count} duplicate rows"
            )
            
        return validation_results
    
    def _preprocess_dataframe(self, df: pd.DataFrame, **options) -> pd.DataFrame:
        """
        Preprocess DataFrame with configurable options.
        
        Args:
            df: DataFrame to preprocess
            **options: Preprocessing options
            
        Returns:
            Preprocessed DataFrame
        """
        processed_df = df.copy()
        
        # Remove duplicates if requested
        if options.get("remove_duplicates", False):
            initial_shape = processed_df.shape
            processed_df = processed_df.drop_duplicates()
            self.logger.info(
                f"Removed {initial_shape[0] - processed_df.shape[0]} duplicate rows"
            )
            
        # Handle missing values
        missing_strategy = options.get("missing_strategy", "none")
        if missing_strategy == "drop":
            processed_df = processed_df.dropna()
        elif missing_strategy == "fill_forward":
            processed_df = processed_df.fillna(method="ffill")
        elif missing_strategy == "fill_backward":
            processed_df = processed_df.fillna(method="bfill")
            
        # Convert data types if specified
        dtype_conversions = options.get("dtype_conversions", {})
        for column, dtype in dtype_conversions.items():
            if column in processed_df.columns:
                try:
                    processed_df[column] = processed_df[column].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Failed to convert {column} to {dtype}: {e}")
                    
        return processed_df
    
    def load_csv(
        self, 
        file_path: str, 
        use_cache: bool = True,
        **pandas_kwargs
    ) -> Dict[str, Any]:
        """
        Load CSV file with caching and validation.
        
        Args:
            file_path: Path to CSV file
            use_cache: Whether to use caching
            **pandas_kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            Dictionary containing DataFrame, validation results, and metadata
        """
        try:
            self.logger.info(f"Loading CSV file: {file_path}")
            
            # Check cache
            if use_cache:
                cache_key = self._generate_cache_key(file_path, **pandas_kwargs)
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                
                if cache_file.exists():
                    self.logger.info("Loading from cache")
                    return pd.read_pickle(cache_file)
            
            # Load CSV
            df = pd.read_csv(file_path, **pandas_kwargs)
            
            # Validate
            validation = self._validate_dataframe(df)
            
            if not validation["is_valid"]:
                raise DataLoadError(f"Data validation failed: {validation['errors']}")
            
            # Preprocess if options provided
            preprocessing_options = pandas_kwargs.pop("preprocessing", {})
            if preprocessing_options:
                df = self._preprocess_dataframe(df, **preprocessing_options)
            
            result = {
                "dataframe": df,
                "validation": validation,
                "file_info": {
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path),
                    "load_method": "csv"
                }
            }
            
            # Cache result
            if use_cache:
                pd.to_pickle(result, cache_file)
                self.logger.info("Result cached successfully")
            
            self.logger.info(f"Successfully loaded CSV with shape {df.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise DataLoadError(f"Failed to load CSV: {str(e)}")
    
    def load_excel(
        self, 
        file_path: str, 
        sheet_name: Union[str, int] = 0,
        use_cache: bool = True,
        **pandas_kwargs
    ) -> Dict[str, Any]:
        """
        Load Excel file with caching and validation.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index to load
            use_cache: Whether to use caching
            **pandas_kwargs: Additional arguments for pandas.read_excel
            
        Returns:
            Dictionary containing DataFrame, validation results, and metadata
        """
        try:
            self.logger.info(f"Loading Excel file: {file_path}, sheet: {sheet_name}")
            
            # Check cache
            cache_params = {"sheet_name": sheet_name, **pandas_kwargs}
            if use_cache:
                cache_key = self._generate_cache_key(file_path, **cache_params)
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                
                if cache_file.exists():
                    self.logger.info("Loading from cache")
                    return pd.read_pickle(cache_file)
            
            # Load Excel
            df = pd.read_excel(file_path, sheet_name=sheet_name, **pandas_kwargs)
            
            # Validate
            validation = self._validate_dataframe(df)
            
            if not validation["is_valid"]:
                raise DataLoadError(f"Data validation failed: {validation['errors']}")
            
            # Preprocess if options provided
            preprocessing_options = pandas_kwargs.pop("preprocessing", {})
            if preprocessing_options:
                df = self._preprocess_dataframe(df, **preprocessing_options)
            
            result = {
                "dataframe": df,
                "validation": validation,
                "file_info": {
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path),
                    "sheet_name": sheet_name,
                    "load_method": "excel"
                }
            }
            
            # Cache result
            if use_cache:
                pd.to_pickle(result, cache_file)
                self.logger.info("Result cached successfully")
            
            self.logger.info(f"Successfully loaded Excel with shape {df.shape}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading Excel file {file_path}: {str(e)}")
            raise DataLoadError(f"Failed to load Excel: {str(e)}")
    
    def load_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Auto-detect file type and load accordingly.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional loading options
            
        Returns:
            Dictionary containing DataFrame, validation results, and metadata
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == ".csv":
            return self.load_csv(file_path, **kwargs)
        elif file_extension in [".xlsx", ".xls"]:
            return self.load_excel(file_path, **kwargs)
        else:
            raise DataLoadError(f"Unsupported file format: {file_extension}")
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive file information without loading.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary containing file metadata
        """
        try:
            file_stat = os.stat(file_path)
            file_path_obj = Path(file_path)
            
            return {
                "file_name": file_path_obj.name,
                "file_size": file_stat.st_size,
                "file_extension": file_path_obj.suffix,
                "modification_time": file_stat.st_mtime,
                "is_readable": os.access(file_path, os.R_OK)
            }
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {str(e)}")
            raise DataLoadError(f"Failed to get file info: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all cached data files."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")


# Factory function for easy instantiation
def create_data_loader(cache_dir: str = "cache") -> DataLoader:
    """
    Factory function to create DataLoader instance.
    
    Args:
        cache_dir: Directory for caching
        
    Returns:
        Configured DataLoader instance
    """
    return DataLoader(cache_dir=cache_dir)
