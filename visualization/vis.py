"""
Visualization module for scalable data agent application.

This module provides comprehensive visualization capabilities with support
for different chart types, export functionality, and modular design for
easy modification and extension.
"""

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import logging
import io
import base64
from typing import Any, Dict, Optional, Union, List, Tuple
from enum import Enum
from pathlib import Path
import json


class ChartType(Enum):
    """Enumeration of supported chart types."""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    AREA = "area"
    VIOLIN = "violin"
    CORRELATION = "correlation"


class VisualizationEngine(Enum):
    """Enumeration of visualization engines."""
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"


class VisualizationError(Exception):
    """Base exception for visualization errors."""
    pass


class ChartConfig:
    """Configuration class for chart settings."""
    
    def __init__(
        self,
        chart_type: ChartType,
        engine: VisualizationEngine = VisualizationEngine.PLOTLY,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        width: int = 800,
        height: int = 600,
        color_scheme: str = "viridis",
        theme: str = "plotly_white",
        **kwargs
    ):
        self.chart_type = chart_type
        self.engine = engine
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.width = width
        self.height = height
        self.color_scheme = color_scheme
        self.theme = theme
        self.custom_options = kwargs


class VisualizationResult:
    """Container for visualization results."""
    
    def __init__(
        self,
        success: bool,
        chart_data: Optional[Any] = None,
        chart_type: Optional[ChartType] = None,
        engine: Optional[VisualizationEngine] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.chart_data = chart_data
        self.chart_type = chart_type
        self.engine = engine
        self.error_message = error_message
        self.metadata = metadata or {}


class BaseVisualizer:
    """Base class for visualization engines."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def create_chart(
        self,
        data: pd.DataFrame,
        config: ChartConfig,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        **kwargs
    ) -> VisualizationResult:
        """Create a chart based on configuration."""
        raise NotImplementedError("Subclasses must implement create_chart")


class PlotlyVisualizer(BaseVisualizer):
    """Plotly-based visualization engine."""
    
    def __init__(self):
        super().__init__()
        
    def create_chart(
        self,
        data: pd.DataFrame,
        config: ChartConfig,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        **kwargs
    ) -> VisualizationResult:
        """Create chart using Plotly."""
        try:
            fig = None
            
            if config.chart_type == ChartType.BAR:
                fig = self._create_bar_chart(data, config, x_col, y_col, **kwargs)
            elif config.chart_type == ChartType.LINE:
                fig = self._create_line_chart(data, config, x_col, y_col, **kwargs)
            elif config.chart_type == ChartType.SCATTER:
                fig = self._create_scatter_chart(data, config, x_col, y_col, **kwargs)
            elif config.chart_type == ChartType.PIE:
                fig = self._create_pie_chart(data, config, x_col, y_col, **kwargs)
            elif config.chart_type == ChartType.HISTOGRAM:
                fig = self._create_histogram(data, config, x_col, **kwargs)
            elif config.chart_type == ChartType.BOX:
                fig = self._create_box_plot(data, config, x_col, y_col, **kwargs)
            elif config.chart_type == ChartType.HEATMAP:
                fig = self._create_heatmap(data, config, **kwargs)
            elif config.chart_type == ChartType.AREA:
                fig = self._create_area_chart(data, config, x_col, y_col, **kwargs)
            elif config.chart_type == ChartType.CORRELATION:
                fig = self._create_correlation_matrix(data, config, **kwargs)
            else:
                raise VisualizationError(f"Unsupported chart type: {config.chart_type}")
            
            if fig:
                # Apply common styling
                fig.update_layout(
                    title=config.title,
                    width=config.width,
                    height=config.height,
                    template=config.theme
                )
                
                if config.x_label:
                    fig.update_xaxes(title_text=config.x_label)
                if config.y_label:
                    fig.update_yaxes(title_text=config.y_label)
                
                return VisualizationResult(
                    success=True,
                    chart_data=fig,
                    chart_type=config.chart_type,
                    engine=VisualizationEngine.PLOTLY,
                    metadata={
                        "data_shape": data.shape,
                        "x_column": x_col,
                        "y_column": y_col
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Error creating Plotly chart: {e}")
            return VisualizationResult(
                success=False,
                error_message=str(e)
            )
    
    def _create_bar_chart(self, data, config, x_col, y_col, **kwargs):
        """Create bar chart."""
        if not x_col or not y_col:
            raise VisualizationError("Bar chart requires both x and y columns")
        
        return px.bar(
            data,
            x=x_col,
            y=y_col,
            color_discrete_sequence=px.colors.qualitative.Set3,
            **kwargs
        )
    
    def _create_line_chart(self, data, config, x_col, y_col, **kwargs):
        """Create line chart."""
        if not x_col or not y_col:
            raise VisualizationError("Line chart requires both x and y columns")
        
        return px.line(
            data,
            x=x_col,
            y=y_col,
            **kwargs
        )
    
    def _create_scatter_chart(self, data, config, x_col, y_col, **kwargs):
        """Create scatter plot."""
        if not x_col or not y_col:
            raise VisualizationError("Scatter plot requires both x and y columns")
        
        return px.scatter(
            data,
            x=x_col,
            y=y_col,
            **kwargs
        )
    
    def _create_pie_chart(self, data, config, x_col, y_col, **kwargs):
        """Create pie chart."""
        if not x_col:
            raise VisualizationError("Pie chart requires values column")
        
        # If y_col is provided, use it as values, otherwise count occurrences
        if y_col:
            return px.pie(data, names=x_col, values=y_col, **kwargs)
        else:
            value_counts = data[x_col].value_counts()
            return px.pie(values=value_counts.values, names=value_counts.index, **kwargs)
    
    def _create_histogram(self, data, config, x_col, **kwargs):
        """Create histogram."""
        if not x_col:
            raise VisualizationError("Histogram requires x column")
        
        return px.histogram(data, x=x_col, **kwargs)
    
    def _create_box_plot(self, data, config, x_col, y_col, **kwargs):
        """Create box plot."""
        if not y_col:
            raise VisualizationError("Box plot requires y column")
        
        return px.box(data, x=x_col, y=y_col, **kwargs)
    
    def _create_heatmap(self, data, config, **kwargs):
        """Create heatmap."""
        # Select only numeric columns
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            raise VisualizationError("No numeric data available for heatmap")
        
        return px.imshow(
            numeric_data.corr(),
            color_continuous_scale=config.color_scheme,
            **kwargs
        )
    
    def _create_area_chart(self, data, config, x_col, y_col, **kwargs):
        """Create area chart."""
        if not x_col or not y_col:
            raise VisualizationError("Area chart requires both x and y columns")
        
        return px.area(data, x=x_col, y=y_col, **kwargs)
    
    def _create_correlation_matrix(self, data, config, **kwargs):
        """Create correlation matrix."""
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.empty:
            raise VisualizationError("No numeric data available for correlation matrix")
        
        corr_matrix = numeric_data.corr()
        
        return px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu",
            aspect="auto",
            **kwargs
        )


class MatplotlibVisualizer(BaseVisualizer):
    """Matplotlib-based visualization engine."""
    
    def __init__(self):
        super().__init__()
        plt.style.use('default')
    
    def create_chart(
        self,
        data: pd.DataFrame,
        config: ChartConfig,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        **kwargs
    ) -> VisualizationResult:
        """Create chart using Matplotlib."""
        try:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            if config.chart_type == ChartType.BAR:
                self._create_bar_chart(data, ax, x_col, y_col, **kwargs)
            elif config.chart_type == ChartType.LINE:
                self._create_line_chart(data, ax, x_col, y_col, **kwargs)
            elif config.chart_type == ChartType.SCATTER:
                self._create_scatter_chart(data, ax, x_col, y_col, **kwargs)
            elif config.chart_type == ChartType.HISTOGRAM:
                self._create_histogram(data, ax, x_col, **kwargs)
            elif config.chart_type == ChartType.BOX:
                self._create_box_plot(data, ax, x_col, y_col, **kwargs)
            else:
                raise VisualizationError(f"Chart type {config.chart_type} not implemented for Matplotlib")
            
            # Apply styling
            ax.set_title(config.title)
            if config.x_label:
                ax.set_xlabel(config.x_label)
            if config.y_label:
                ax.set_ylabel(config.y_label)
            
            plt.tight_layout()
            
            return VisualizationResult(
                success=True,
                chart_data=fig,
                chart_type=config.chart_type,
                engine=VisualizationEngine.MATPLOTLIB,
                metadata={
                    "data_shape": data.shape,
                    "x_column": x_col,
                    "y_column": y_col
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error creating Matplotlib chart: {e}")
            return VisualizationResult(
                success=False,
                error_message=str(e)
            )
    
    def _create_bar_chart(self, data, ax, x_col, y_col, **kwargs):
        """Create bar chart."""
        if not x_col or not y_col:
            raise VisualizationError("Bar chart requires both x and y columns")
        
        ax.bar(data[x_col], data[y_col], **kwargs)
    
    def _create_line_chart(self, data, ax, x_col, y_col, **kwargs):
        """Create line chart."""
        if not x_col or not y_col:
            raise VisualizationError("Line chart requires both x and y columns")
        
        ax.plot(data[x_col], data[y_col], **kwargs)
    
    def _create_scatter_chart(self, data, ax, x_col, y_col, **kwargs):
        """Create scatter plot."""
        if not x_col or not y_col:
            raise VisualizationError("Scatter plot requires both x and y columns")
        
        ax.scatter(data[x_col], data[y_col], **kwargs)
    
    def _create_histogram(self, data, ax, x_col, **kwargs):
        """Create histogram."""
        if not x_col:
            raise VisualizationError("Histogram requires x column")
        
        ax.hist(data[x_col], **kwargs)
    
    def _create_box_plot(self, data, ax, x_col, y_col, **kwargs):
        """Create box plot."""
        if not y_col:
            raise VisualizationError("Box plot requires y column")
        
        if x_col:
            groups = data.groupby(x_col)[y_col].apply(list)
            ax.boxplot(groups.values, labels=groups.index, **kwargs)
        else:
            ax.boxplot(data[y_col], **kwargs)


class VisualizationManager:
    """
    Central visualization management system.
    
    This class provides a unified interface for different visualization
    engines and chart types with export capabilities.
    """
    
    def __init__(self):
        self.engines = {
            VisualizationEngine.PLOTLY: PlotlyVisualizer(),
            VisualizationEngine.MATPLOTLIB: MatplotlibVisualizer()
        }
        self.default_engine = VisualizationEngine.PLOTLY
        self.logger = logging.getLogger(__name__)
    
    def create_chart(
        self,
        data: pd.DataFrame,
        chart_type: ChartType,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        engine: Optional[VisualizationEngine] = None,
        config: Optional[ChartConfig] = None,
        **kwargs
    ) -> VisualizationResult:
        """
        Create a chart with specified parameters.
        
        Args:
            data: DataFrame containing the data
            chart_type: Type of chart to create
            x_col: Column for x-axis
            y_col: Column for y-axis
            engine: Visualization engine to use
            config: Chart configuration
            **kwargs: Additional chart options
            
        Returns:
            VisualizationResult object
        """
        try:
            # Use provided engine or default
            engine = engine or self.default_engine
            
            # Create config if not provided
            if not config:
                config = ChartConfig(
                    chart_type=chart_type,
                    engine=engine,
                    **kwargs
                )
            
            # Get visualizer
            visualizer = self.engines[engine]
            
            # Create chart
            result = visualizer.create_chart(data, config, x_col, y_col, **kwargs)
            
            if result.success:
                self.logger.info(f"Chart created successfully: {chart_type.value}")
            else:
                self.logger.error(f"Chart creation failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in chart creation: {e}")
            return VisualizationResult(
                success=False,
                error_message=str(e)
            )
    
    def auto_visualize(self, data: pd.DataFrame, column: Optional[str] = None) -> List[VisualizationResult]:
        """
        Automatically create appropriate visualizations for the data.
        
        Args:
            data: DataFrame to visualize
            column: Specific column to focus on (optional)
            
        Returns:
            List of VisualizationResult objects
        """
        results = []
        
        try:
            if column and column in data.columns:
                # Visualize specific column
                results.extend(self._visualize_column(data, column))
            else:
                # Visualize entire dataset
                results.extend(self._visualize_dataset(data))
                
        except Exception as e:
            self.logger.error(f"Error in auto visualization: {e}")
            results.append(VisualizationResult(
                success=False,
                error_message=str(e)
            ))
        
        return results
    
    def _visualize_column(self, data: pd.DataFrame, column: str) -> List[VisualizationResult]:
        """Create appropriate visualizations for a specific column."""
        results = []
        
        col_data = data[column]
        
        # Determine column type and create appropriate visualizations
        if col_data.dtype in ['int64', 'float64']:
            # Numeric column
            results.append(self.create_chart(
                data, ChartType.HISTOGRAM, x_col=column,
                title=f"Distribution of {column}"
            ))
            
            if len(data.select_dtypes(include=['number']).columns) > 1:
                results.append(self.create_chart(
                    data, ChartType.BOX, y_col=column,
                    title=f"Box Plot of {column}"
                ))
        
        elif col_data.dtype == 'object':
            # Categorical column
            if col_data.nunique() <= 20:  # Reasonable number of categories
                results.append(self.create_chart(
                    data, ChartType.PIE, x_col=column,
                    title=f"Distribution of {column}"
                ))
        
        return results
    
    def _visualize_dataset(self, data: pd.DataFrame) -> List[VisualizationResult]:
        """Create overview visualizations for the entire dataset."""
        results = []
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Correlation matrix for numeric data
        if len(numeric_cols) > 1:
            results.append(self.create_chart(
                data, ChartType.CORRELATION,
                title="Correlation Matrix"
            ))
        
        # Distribution of first few numeric columns
        for col in numeric_cols[:3]:
            results.append(self.create_chart(
                data, ChartType.HISTOGRAM, x_col=col,
                title=f"Distribution of {col}"
            ))
        
        # Pie charts for categorical columns with reasonable cardinality
        for col in categorical_cols[:2]:
            if data[col].nunique() <= 10:
                results.append(self.create_chart(
                    data, ChartType.PIE, x_col=col,
                    title=f"Distribution of {col}"
                ))
        
        return results
    
    def export_chart(
        self,
        result: VisualizationResult,
        file_path: str,
        format: str = "png"
    ) -> bool:
        """
        Export chart to file.
        
        Args:
            result: VisualizationResult containing chart data
            file_path: Path to save the file
            format: Export format ('png', 'html', 'svg', etc.)
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            if not result.success or not result.chart_data:
                self.logger.error("Cannot export failed chart result")
                return False
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if result.engine == VisualizationEngine.PLOTLY:
                if format.lower() == "html":
                    result.chart_data.write_html(str(file_path))
                elif format.lower() in ["png", "jpg", "jpeg", "svg", "pdf"]:
                    result.chart_data.write_image(str(file_path))
                else:
                    raise VisualizationError(f"Unsupported export format: {format}")
            
            elif result.engine == VisualizationEngine.MATPLOTLIB:
                result.chart_data.savefig(str(file_path), format=format, dpi=300, bbox_inches='tight')
                plt.close(result.chart_data)
            
            self.logger.info(f"Chart exported successfully to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting chart: {e}")
            return False


# Factory function for easy instantiation
def create_visualization_manager() -> VisualizationManager:
    """
    Factory function to create VisualizationManager instance.
    
    Returns:
        Configured VisualizationManager instance
    """
    return VisualizationManager()


# Convenience functions
def create_chart(
    data: pd.DataFrame,
    chart_type: str,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    **kwargs
) -> VisualizationResult:
    """
    Convenience function to create a chart.
    
    Args:
        data: DataFrame containing the data
        chart_type: Type of chart to create (string)
        x_col: Column for x-axis
        y_col: Column for y-axis
        **kwargs: Additional chart options
        
    Returns:
        VisualizationResult object
    """
    manager = create_visualization_manager()
    chart_type_enum = ChartType(chart_type.lower())
    return manager.create_chart(data, chart_type_enum, x_col, y_col, **kwargs)
