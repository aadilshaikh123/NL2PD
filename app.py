"""
Main Streamlit application for scalable data agent.

This application provides a modular, scalable interface for natural language
data analysis using pandas-ai with Groq LLM backend. Designed for easy
migration to React/backend architecture.
"""

import streamlit as st
import pandas as pd
import logging
import os
import time
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import traceback

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from core.data_loader import create_data_loader, DataLoadError
from core.llm_manager import create_llm_manager_from_env, LLMError
from core.query_processor import create_query_processor_manager, QueryProcessorError
from visualization.vis import create_visualization_manager, ChartType, VisualizationEngine
from storage.history_manager import create_history_manager

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class DataAgentApp:
    """
    Main application class for the scalable data agent.
    
    This class orchestrates all components and provides a clean interface
    for easy migration to backend API architecture.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.data_loader = None
        self.llm_manager = None
        self.query_processor = None
        self.visualization_manager = None
        self.history_manager = None
        self.logger = None
        
        # Initialize components
        self._setup_logging()
        self._initialize_components()
    
    def _setup_logging(self) -> None:
        """Setup comprehensive logging."""
        try:
            # Create logs directory
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Configure logging with UTF-8 encoding
            log_file = logs_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8'),
                    logging.StreamHandler()
                ],
                force=True  # Override any existing configuration
            )
            
            self.logger = logging.getLogger(__name__)
            self.logger.info("Logging initialized successfully")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO, force=True)
            self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self) -> None:
        """Initialize all application components."""
        try:
            self.logger.info("Initializing application components...")
            
            # Initialize data loader
            self.data_loader = create_data_loader()
            
            # Initialize visualization manager
            self.visualization_manager = create_visualization_manager()
            
            # Initialize history manager
            self.history_manager = create_history_manager()
            
            self.logger.info("Core components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            st.error(f"Failed to initialize application components: {e}")
    
    def _initialize_llm_components(self, api_key: str) -> bool:
        """
        Initialize LLM-dependent components.
        
        Args:
            api_key: Groq API key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not api_key or api_key == "your-groq-api-key-goes-here":
                return False
            
            # Set environment variable
            os.environ["GROQ_API_KEY"] = api_key
            
            # Initialize LLM manager
            self.llm_manager = create_llm_manager_from_env()
            
            # Initialize query processor
            self.query_processor = create_query_processor_manager(self.llm_manager)
            
            self.logger.info("LLM components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing LLM components: {e}")
            st.error(f"Failed to initialize LLM components: {e}")
            return False
    
    def render_sidebar(self) -> Dict[str, Any]:
        """
        Render sidebar with configuration options.
        
        Returns:
            Dictionary containing sidebar state
        """
        st.sidebar.title("🤖 Data Agent Configuration")
        
        # Try to get API key from environment first
        env_api_key = os.getenv("GROQ_API_KEY")
        
        # API Key input - show current status if loaded from env
        if env_api_key and env_api_key != "your-groq-api-key-goes-here":
            st.sidebar.success("✅ API Key loaded from environment")
            st.sidebar.text("Using API key from .env file")
            api_key = env_api_key
        else:
            api_key = st.sidebar.text_input(
                "Groq API Key",
                type="password",
                help="Enter your Groq API key to enable AI features"
            )
        
        # File uploader
        uploaded_file = st.sidebar.file_uploader(
            "Upload Data File",
            type=["csv", "xlsx", "xls"],
            help="Upload a CSV or Excel file for analysis"
        )
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        
        show_debug = st.sidebar.checkbox("Show Debug Info", value=False)
        
        # History controls
        st.sidebar.subheader("📊 History")
        
        show_history = st.sidebar.checkbox("Show Query History", value=False)
        
        if st.sidebar.button("Clear History"):
            if st.sidebar.checkbox("Confirm Clear", key="confirm_clear"):
                self.history_manager.clear_history(confirm=True)
                st.sidebar.success("History cleared!")
                st.experimental_rerun()
        
        # Sample data option
        if st.sidebar.button("Load Sample Data"):
            return {
                "api_key": api_key,
                "uploaded_file": "sample",
                "show_debug": show_debug,
                "show_history": show_history
            }
        
        return {
            "api_key": api_key,
            "uploaded_file": uploaded_file,
            "show_debug": show_debug,
            "show_history": show_history
        }
    
    def load_data(self, file_source: Union[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load data from file or sample.
        
        Args:
            file_source: File object or "sample" string
            
        Returns:
            Loaded DataFrame or None
        """
        try:
            if file_source == "sample":
                # Load sample data
                sample_path = Path("data/sample_student_data.csv")
                if sample_path.exists():
                    result = self.data_loader.load_csv(str(sample_path))
                    if result and "dataframe" in result:
                        st.success("Sample data loaded successfully!")
                        return result["dataframe"]
                else:
                    st.error("Sample data file not found")
                    return None
            
            elif file_source is not None:
                # Load uploaded file
                # Save uploaded file temporarily
                temp_path = Path("temp") / file_source.name
                temp_path.parent.mkdir(exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(file_source.getbuffer())
                
                # Load data
                result = self.data_loader.load_file(str(temp_path))
                
                # Clean up temp file
                temp_path.unlink()
                
                if result and "dataframe" in result:
                    st.success(f"File '{file_source.name}' loaded successfully!")
                    return result["dataframe"]
            
            return None
            
        except DataLoadError as e:
            st.error(f"Data loading error: {e}")
            self.logger.error(f"Data loading error: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error loading data: {e}")
            self.logger.error(f"Unexpected error loading data: {e}")
            return None
    
    def process_query(self, query: str, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Process natural language query.
        
        Args:
            query: User query
            dataframe: DataFrame to query
            
        Returns:
            Dictionary containing result and metadata
        """
        try:
            if not self.query_processor:
                return {
                    "success": False,
                    "error": "Query processor not initialized. Please check your API key."
                }
            
            start_time = time.time()
            
            # Process query
            result = self.query_processor.process_query(query, dataframe)
            
            execution_time = time.time() - start_time
            
            # Add to history
            if result.success:
                # Convert result to string for history (exclude visualization data)
                if hasattr(result.result, 'to_string'):
                    result_str = result.result.to_string()
                elif isinstance(result.result, str):
                    result_str = result.result
                else:
                    result_str = str(result.result)
                
                self.history_manager.add_entry(
                    input_text=query,
                    result=result_str,
                    execution_time=execution_time,
                    error=result.error_message
                )
            else:
                self.history_manager.add_entry(
                    input_text=query,
                    result="",
                    execution_time=execution_time,
                    error=result.error_message
                )
            
            return {
                "success": result.success,
                "result": result.result,
                "result_type": result.result_type,
                "execution_time": execution_time,
                "error": result.error_message,
                "metadata": result.metadata
            }
            
        except QueryProcessorError as e:
            error_msg = f"Query processing error: {e}"
            st.error(error_msg)
            self.logger.error(error_msg)
            
            self.history_manager.add_entry(
                input_text=query,
                result="",
                error=str(e)
            )
            
            return {"success": False, "error": str(e)}
            
        except Exception as e:
            error_msg = f"Unexpected error processing query: {e}"
            st.error(error_msg)
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            self.history_manager.add_entry(
                input_text=query,
                result="",
                error=str(e)
            )
            
            return {"success": False, "error": str(e)}
    
    def display_results(
        self,
        result: Any,
        result_type: str,
        show_visualizations: bool = True
    ) -> None:
        """
        Display query results intelligently.
        
        Args:
            result: Query result
            result_type: Type of result
            show_visualizations: Whether to show visualizations
        """
        try:
            if result_type == "dataframe":
                st.subheader("📊 Query Result")
                st.dataframe(result, use_container_width=True)
                    
            elif result_type == "chart":
                st.subheader("📈 Generated Chart")
                if isinstance(result, str) and any(ext in result for ext in ['.png', '.jpg', '.jpeg']):
                    st.image(result, caption="Generated Chart")
                else:
                    st.write(result)
                    
            elif result_type == "text":
                st.subheader("💬 AI Response")
                st.write(result)
                
            else:
                st.subheader("📄 Result")
                st.write(result)
                
        except Exception as e:
            st.error(f"Error displaying results: {e}")
            self.logger.error(f"Error displaying results: {e}")
    
    def render_history(self) -> None:
        """Render query history interface."""
        try:
            st.subheader("📊 Query History")
            
            # Get recent entries
            entries = self.history_manager.get_recent_entries(limit=20)
            
            if not entries:
                st.info("No query history available.")
                return
            
            # Display entries
            for entry in reversed(entries[-10:]):  # Show last 10 entries
                with st.expander(f"🕒 {entry.timestamp} - {entry.input[:50]}..."):
                    st.write(f"**Query:** {entry.input}")
                    st.write(f"**Result:** {entry.result[:500]}...")
                    
                    if entry.error:
                        st.error(f"**Error:** {entry.error}")
                    
                    if entry.execution_time:
                        st.info(f"**Execution Time:** {entry.execution_time:.2f}s")
            
            # Statistics
            stats = self.history_manager.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Queries", stats.get("total_entries", 0))
            with col2:
                st.metric("Error Rate", f"{stats.get('error_rate', 0)*100:.1f}%")
            with col3:
                avg_time = stats.get("average_execution_time")
                st.metric("Avg Time", f"{avg_time:.2f}s" if avg_time else "N/A")
                
        except Exception as e:
            st.error(f"Error displaying history: {e}")
            self.logger.error(f"Error displaying history: {e}")
    
    def render_main_interface(self, sidebar_state: Dict[str, Any]) -> None:
        """
        Render main application interface.
        
        Args:
            sidebar_state: State from sidebar
        """
        # Main title
        st.title("🤖 AI Data Agent")
        st.markdown("*Ask questions about your data in natural language*")
        
        # Check API key
        if not sidebar_state["api_key"]:
            st.warning("⚠️ Please enter your Groq API key in the sidebar to enable AI features.")
            st.info("You can get a free API key from [Groq Console](https://console.groq.com/)")
            return
        
        # Initialize LLM components if needed
        if not self.llm_manager:
            with st.spinner("Initializing AI components..."):
                if not self._initialize_llm_components(sidebar_state["api_key"]):
                    st.error("Failed to initialize AI components. Please check your API key.")
                    return
        
        # Load data
        dataframe = None
        if sidebar_state["uploaded_file"]:
            with st.spinner("Loading data..."):
                dataframe = self.load_data(sidebar_state["uploaded_file"])
        
        if dataframe is not None:
            # Store in session state
            st.session_state.dataframe = dataframe
            
            # Data preview
            st.subheader("📋 Data Preview")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(dataframe.head(), use_container_width=True)
            with col2:
                st.info(f"**Shape:** {dataframe.shape[0]} rows × {dataframe.shape[1]} columns")
                st.info(f"**Columns:** {', '.join(dataframe.columns.tolist()[:3])}{'...' if len(dataframe.columns) > 3 else ''}")
            
            # Query interface
            st.subheader("💬 Ask Questions About Your Data")
            
            # Example queries
            with st.expander("💡 Example Queries"):
                example_queries = [
                    "What is the average stipend by department?",
                    "Show me the distribution of students by gender",
                    "Which department has the highest average stipend?",
                    "Create a bar chart of student count by department",
                    "What is the age distribution of students?",
                    "Show correlation between age and stipend"
                ]
                
                for query in example_queries:
                    if st.button(f"💭 {query}", key=f"example_{hash(query)}"):
                        st.session_state.example_query = query
            
            # Query input
            query = st.text_area(
                "Enter your question:",
                value=getattr(st.session_state, 'example_query', ''),
                height=100,
                help="Ask questions in natural language about your data"
            )
            
            # Output type selection
            st.subheader("📊 Output Options")
            output_type = st.radio(
                "Choose output type:",
                ["Result Only", "Visualization Only"],
                horizontal=True,
                help="Select what to generate after analysis"
            )
            
            # Chart type selection (only show if visualization is selected)
            chart_type = None
            if output_type == "Visualization Only":
                chart_type = st.selectbox(
                    "Select chart type:",
                    [
                        "Auto-select",
                        "Bar Chart", 
                        "Line Chart",
                        "Scatter Plot",
                        "Pie Chart", 
                        "Histogram",
                        "Box Plot",
                        "Heatmap",
                        "Area Chart"
                    ],
                    help="Choose the type of visualization to generate"
                )
            
            # Query controls
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🚀 Analyze", type="primary", disabled=not query.strip()):
                    # Clear any existing session state to avoid conflicts
                    for key in ['query_result', 'visualization_result', 'view_mode']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    with st.spinner("🤔 AI is analyzing your data..."):
                        if output_type == "Result Only":
                            # Process query for result only
                            result = self.process_query(query, dataframe)
                            
                            if result["success"]:
                                st.success("✅ Analysis completed!")
                                st.session_state.query_result = result
                                
                                # Display result with improved formatting
                                self.display_formatted_results(
                                    result["result"],
                                    result["result_type"]
                                )
                            else:
                                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        
                        elif output_type == "Visualization Only":
                            # Generate visualization based on query and chart type
                            viz_result = self.generate_custom_visualization(
                                query, dataframe, chart_type
                            )
                            
                            if viz_result["success"]:
                                st.success("✅ Visualization generated!")
                                st.session_state.visualization_result = viz_result
                                
                                # Display visualization
                                self.display_visualization(viz_result)
                            else:
                                st.error(f"❌ Visualization failed: {viz_result.get('error', 'Unknown error')}")
            
            with col2:
                if st.button("🗑️ Clear", key="clear_query_btn"):
                    # Clear all session state related to queries
                    keys_to_clear = ['example_query', 'query_result', 'visualization_result', 'view_mode']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.experimental_rerun()
            
            # Debug info
            if sidebar_state["show_debug"]:
                st.subheader("🔧 Debug Information")
                st.json({
                    "DataFrame Shape": dataframe.shape,
                    "DataFrame Columns": dataframe.columns.tolist(),
                    "DataFrame Types": dataframe.dtypes.to_dict(),
                    "Session State Keys": list(st.session_state.keys())
                })
        
        # History section
        if sidebar_state["show_history"]:
            self.render_history()
    
    def display_formatted_results(
        self,
        result: Any,
        result_type: str
    ) -> None:
        """
        Display query results with improved formatting.
        
        Args:
            result: Query result
            result_type: Type of result
        """
        try:
            st.subheader("💬 Analysis Result")
            
            if result_type == "dataframe":
                # Format DataFrame display
                if isinstance(result, pd.DataFrame):
                    if len(result) > 100:
                        st.info(f"📊 Showing first 100 rows of {len(result)} total rows")
                        st.dataframe(result.head(100), use_container_width=True)
                    else:
                        st.dataframe(result, use_container_width=True)
                    
                    # Add summary info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows", len(result))
                    with col2:
                        st.metric("Columns", len(result.columns))
                    with col3:
                        memory_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
                        st.metric("Memory", f"{memory_mb:.2f} MB")
                else:
                    st.dataframe(result, use_container_width=True)
                    
            elif result_type == "text":
                # Format text results with better styling
                if isinstance(result, str):
                    # Handle long text
                    if len(result) > 2000:
                        with st.expander("📄 Full Result (Click to expand)"):
                            st.markdown(result)
                        # Show summary
                        lines = result.split('\n')
                        summary_lines = lines[:10]
                        if len(lines) > 10:
                            summary_lines.append(f"... and {len(lines) - 10} more lines")
                        st.markdown('\n'.join(summary_lines))
                    else:
                        # Format based on content type
                        if '\n' in result and ('|' in result or result.count('\n') > 5):
                            # Looks like structured data
                            st.markdown("```")
                            st.text(result)
                            st.markdown("```")
                        else:
                            # Regular text
                            st.markdown(result)
                else:
                    st.write(result)
                    
            elif result_type == "chart":
                st.subheader("📈 Generated Chart")
                if isinstance(result, str) and any(ext in result for ext in ['.png', '.jpg', '.jpeg']):
                    st.image(result, caption="Generated Chart")
                else:
                    st.plotly_chart(result, use_container_width=True)
                    
            else:
                # Handle other result types
                if isinstance(result, (list, dict)):
                    st.json(result)
                elif isinstance(result, (int, float)):
                    st.metric("Result", result)
                else:
                    st.write(result)
                
        except Exception as e:
            st.error(f"Error displaying results: {e}")
            self.logger.error(f"Error displaying formatted results: {e}")
    
    def generate_custom_visualization(
        self,
        query: str,
        dataframe: pd.DataFrame,
        chart_type: str
    ) -> Dict[str, Any]:
        """
        Generate visualization based on query and selected chart type.
        
        Args:
            query: User query for context
            dataframe: DataFrame to visualize
            chart_type: Selected chart type
            
        Returns:
            Dictionary containing visualization result
        """
        try:
            from visualization.vis import ChartType
            
            # Map string chart types to enum values
            chart_mapping = {
                "Auto-select": None,
                "Bar Chart": ChartType.BAR,
                "Line Chart": ChartType.LINE,
                "Scatter Plot": ChartType.SCATTER,
                "Pie Chart": ChartType.PIE,
                "Histogram": ChartType.HISTOGRAM,
                "Box Plot": ChartType.BOX,
                "Heatmap": ChartType.HEATMAP,
                "Area Chart": ChartType.AREA
            }
            
            selected_chart_type = chart_mapping.get(chart_type)
            
            if selected_chart_type is None:
                # Auto-select based on data
                viz_results = self.visualization_manager.auto_visualize(dataframe)
                if viz_results and viz_results[0].success:
                    return {
                        "success": True,
                        "chart": viz_results[0].chart_data,
                        "chart_type": "auto-selected",
                        "metadata": viz_results[0].metadata
                    }
                else:
                    return {"success": False, "error": "Auto-visualization failed"}
            else:
                # Generate specific chart type
                viz_result = self.visualization_manager.create_chart(
                    dataframe,
                    chart_type=selected_chart_type,
                    title=f"Visualization for: {query[:50]}...",
                    auto_detect_columns=True
                )
                
                if viz_result.success:
                    return {
                        "success": True,
                        "chart": viz_result.chart_data,
                        "chart_type": chart_type,
                        "metadata": viz_result.metadata
                    }
                else:
                    return {
                        "success": False, 
                        "error": viz_result.error_message or "Visualization generation failed"
                    }
                    
        except Exception as e:
            self.logger.error(f"Error generating custom visualization: {e}")
            return {"success": False, "error": str(e)}
    
    def display_visualization(self, viz_result: Dict[str, Any]) -> None:
        """
        Display visualization result.
        
        Args:
            viz_result: Visualization result dictionary
        """
        try:
            st.subheader(f"📈 {viz_result.get('chart_type', 'Chart').title()}")
            
            chart = viz_result.get("chart")
            if chart is not None:
                st.plotly_chart(chart, use_container_width=True)
                
                # Show metadata if available
                metadata = viz_result.get("metadata", {})
                if metadata:
                    with st.expander("📊 Chart Details"):
                        st.json(metadata)
            else:
                st.error("Chart data not available")
                
        except Exception as e:
            st.error(f"Error displaying visualization: {e}")
            self.logger.error(f"Error displaying visualization: {e}")
    
    def run(self) -> None:
        """Run the main application."""
        try:
            # Page configuration
            st.set_page_config(
                page_title="AI Data Agent",
                page_icon="🤖",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Custom CSS
            st.markdown("""
            <style>
            .main {
                padding-top: 1rem;
            }
            .stButton > button {
                width: 100%;
            }
            .metric-container {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Render sidebar
            sidebar_state = self.render_sidebar()
            
            # Render main interface
            self.render_main_interface(sidebar_state)
            
            # Footer
            st.markdown("---")
            st.markdown(
                "🚀 **Scalable Data Agent** | Built with Streamlit, pandas-ai, and Groq | "
                "Ready for React/Backend migration"
            )
            
        except Exception as e:
            st.error(f"Application error: {e}")
            self.logger.error(f"Application error: {e}\n{traceback.format_exc()}")


def main():
    """Main application entry point."""
    try:
        app = DataAgentApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to start application: {e}")
        print(f"Failed to start application: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
