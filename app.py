import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import traceback
import os
import urllib3
import certifi
from utils.data_handler import DataHandler
from utils.query_processor import QueryProcessor
from utils.groq_client import GroqClient

# Global flag to prevent multiple SSL setups
_ssl_setup_done = False

# Fix SSL certificate issues at startup
def fix_ssl_environment():
    """Fix SSL environment variables to handle certificate issues"""
    global _ssl_setup_done
    
    if _ssl_setup_done:
        return
    
    import sys
    
    # Disable SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Aggressively remove problematic environment variables
    ssl_vars_to_remove = ['SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'SSL_CERT_DIR']
    
    for var in ssl_vars_to_remove:
        if var in os.environ:
            old_value = os.environ[var]
            del os.environ[var]
            print(f"Removed {var}: {old_value}")
    
    # Set proper certificate bundle
    cert_path = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    os.environ['SSL_CERT_FILE'] = cert_path
    
    print(f"SSL certificate bundle set to: {cert_path}")
    
    _ssl_setup_done = True

# Fix SSL issues before doing anything else
fix_ssl_environment()

# Page configuration
st.set_page_config(
    page_title="Natural Language to Pandas Query Converter",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🐼 Natural Language to Pandas Query Converter")
    st.markdown("Upload your CSV/XLS file and query it using natural language!")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'groq_client' not in st.session_state:
        try:
            st.session_state.groq_client = GroqClient()
        except Exception as e:
            st.error(f"❌ Failed to initialize Groq client: {str(e)}")
            st.info("💡 Please check your GROQ_API_KEY environment variable")
            st.session_state.groq_client = None
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = QueryProcessor()
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your data file to start querying"
        )
        
        if uploaded_file is not None:
            try:
                # Load data using DataHandler
                data_handler = DataHandler()
                st.session_state.df = data_handler.load_file(uploaded_file)
                st.success(f"✅ File loaded successfully!")
                st.info(f"Shape: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
                
                # Display basic info about the dataset
                st.subheader("📊 Dataset Info")
                st.write(f"**Columns:** {list(st.session_state.df.columns)}")
                st.write(f"**Data Types:**")
                for col, dtype in st.session_state.df.dtypes.items():
                    st.write(f"- {col}: {dtype}")
                    
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.session_state.df = None
        
        # API Key status
        st.header("🔑 API Configuration")
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if groq_api_key:
            st.success("✅ Groq API Key found")
        else:
            st.warning("⚠️ Groq API Key not found in environment variables")
    
    # Main content area
    if st.session_state.df is not None:
        # Data preview
        st.header("📋 Data Preview")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(st.session_state.df.head(10), use_container_width=True)
        with col2:
            st.metric("Total Rows", st.session_state.df.shape[0])
            st.metric("Total Columns", st.session_state.df.shape[1])
            st.metric("Memory Usage", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Query interface
        st.header("💬 Natural Language Query")
        
        # Sample queries
        with st.expander("💡 Sample Queries"):
            sample_queries = [
                "Show me the first 10 rows",
                "What are the column names?",
                "Calculate the mean of all numeric columns",
                "Show me the data types of each column",
                "Find rows where [column_name] is greater than [value]",
                "Group by [column_name] and calculate the sum",
                "Show me the correlation between numeric columns",
                "Find missing values in the dataset",
                "Sort by [column_name] in descending order",
                "Create a histogram of [column_name]"
            ]
            for query in sample_queries:
                if st.button(query, key=f"sample_{query}"):
                    st.session_state.current_query = query
        
        # Query input
        query = st.text_area(
            "Enter your query in natural language:",
            value=st.session_state.get('current_query', ''),
            height=100,
            placeholder="e.g., 'Show me the average sales by region' or 'Find all customers with age > 30'"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("🚀 Execute Query", type="primary"):
                if query.strip():
                    execute_query(query)
                else:
                    st.warning("Please enter a query first!")
        
        with col2:
            if st.button("🗑️ Clear Results"):
                if 'query_results' in st.session_state:
                    del st.session_state.query_results
                if 'generated_code' in st.session_state:
                    del st.session_state.generated_code
                st.rerun()
        
        # Display results
        if 'query_results' in st.session_state and 'generated_code' in st.session_state:
            display_results()
        
        # Query history
        if st.session_state.query_history:
            st.header("📜 Query History")
            for i, history_item in enumerate(reversed(st.session_state.query_history[-5:])):
                # Handle both old format (query, timestamp) and new format (query, timestamp, execution_time)
                if len(history_item) == 3:
                    hist_query, timestamp, execution_time = history_item
                    exec_time_display = f" ⏱️ {execution_time}s"
                else:
                    hist_query, timestamp = history_item
                    exec_time_display = ""
                
                with st.expander(f"Query {len(st.session_state.query_history) - i}: {hist_query[:50]}...{exec_time_display}"):
                    st.write(f"**Query:** {hist_query}")
                    st.write(f"**Timestamp:** {timestamp}")
                    if exec_time_display:
                        st.write(f"**Execution Time:** {execution_time}s")
                    if st.button(f"Re-run this query", key=f"rerun_{i}"):
                        execute_query(hist_query)
            
            # Performance Statistics
            execution_times = [item[2] for item in st.session_state.query_history if len(item) == 3]
            if execution_times:
                st.subheader("⚡ Performance Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Time", f"{sum(execution_times)/len(execution_times):.2f}s")
                with col2:
                    st.metric("Fastest Query", f"{min(execution_times):.2f}s")
                with col3:
                    st.metric("Slowest Query", f"{max(execution_times):.2f}s")
                with col4:
                    st.metric("Total Queries", len(execution_times))
    
    else:
        # Welcome screen
        st.header("👋 Welcome!")
        st.markdown("""
        **Get started by uploading your data file:**
        
        1. 📤 Upload a CSV or Excel file using the sidebar
        2. 👀 Preview your data to understand its structure
        3. 💬 Ask questions in natural language
        4. 📊 Get pandas code and results instantly!
        
        **Supported file formats:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        
        **Example queries you can try:**
        - "Show me the first 10 rows"
        - "What's the average of column X?"
        - "Find all rows where column Y is greater than 100"
        - "Create a chart showing the distribution of column Z"
        """)

def execute_query(query):
    """Execute a natural language query"""
    try:
        # Check if Groq client is available
        if st.session_state.groq_client is None:
            st.error("❌ Groq client not available. Please check your API key configuration.")
            return
            
        with st.spinner("🤖 Processing your query..."):
            # Start timing
            import datetime
            import time
            start_time = time.time()
            
            # Process query using PandasAI with Groq
            result = st.session_state.query_processor.process_query(
                st.session_state.df, 
                query, 
                st.session_state.groq_client
            )
            
            # Calculate execution time
            end_time = time.time()
            execution_time = round(end_time - start_time, 2)
            
            # Add to history with execution time
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.query_history.append((query, timestamp, execution_time))
            
            if result['success']:
                st.session_state.query_results = result['result']
                st.session_state.generated_code = result['code']
                st.success(f"✅ Query executed successfully! ⏱️ Execution time: {execution_time}s")
            else:
                st.error(f"❌ Error executing query: {result['error']}")
                
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.code(traceback.format_exc())

def display_results():
    """Display query results with generated code"""
    st.header("📊 Query Results")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🎯 Results", "💻 Generated Code", "📈 Visualization"])
    
    with tab1:
        st.subheader("Query Output")
        result = st.session_state.query_results
        
        if isinstance(result, pd.DataFrame):
            if not result.empty:
                st.dataframe(result, use_container_width=True)
                
                # Download option
                csv = result.to_csv(index=False)
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("Query returned an empty result set.")
        else:
            st.write(result)
    
    with tab2:
        st.subheader("Generated Pandas Code")
        st.code(st.session_state.generated_code, language="python")
        
        # Copy to clipboard button simulation
        if st.button("📋 Copy Code"):
            st.success("Code copied to clipboard! (Use Ctrl+C to copy from the code block above)")
    
    with tab3:
        st.subheader("Data Visualization")
        result = st.session_state.query_results
        
        if isinstance(result, pd.DataFrame) and not result.empty:
            # Auto-generate visualizations based on data type
            numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = result.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_cols) > 0:
                st.write("**Numeric Data Visualization:**")
                
                if len(numeric_cols) == 1:
                    # Single numeric column - histogram
                    fig = px.histogram(result, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
                    st.plotly_chart(fig, use_container_width=True)
                elif len(numeric_cols) >= 2:
                    # Multiple numeric columns - scatter plot of first two
                    fig = px.scatter(result, x=numeric_cols[0], y=numeric_cols[1], 
                                   title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                    st.plotly_chart(fig, use_container_width=True)
            
            if len(categorical_cols) > 0 and len(result) <= 1000:  # Limit for performance
                st.write("**Categorical Data Visualization:**")
                for col in categorical_cols[:2]:  # Limit to first 2 categorical columns
                    value_counts = result[col].value_counts().head(10)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, 
                               title=f"Top 10 values in {col}")
                    fig.update_layout(xaxis_title=col, yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No visualization available for this result type.")

if __name__ == "__main__":
    main()
