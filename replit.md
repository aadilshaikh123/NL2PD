# Natural Language to Pandas Query Converter

## Overview

This is a Streamlit-based web application that allows users to upload CSV or Excel files and query their data using natural language. The application leverages the Groq API with Llama3-70b-8192 model to convert natural language queries into executable pandas code, making data analysis accessible to users without programming knowledge.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid UI development
- **Layout**: Wide layout with expandable sidebar for file upload and configuration
- **State Management**: Session state management for DataFrame persistence, query history, and client instances
- **File Upload**: Built-in Streamlit file uploader supporting CSV, XLS, and XLSX formats

### Backend Architecture
- **Modular Design**: Utility classes organized in separate modules for maintainability
- **Data Handler**: `DataHandler` class manages file loading with automatic delimiter detection and multiple encoding support
- **Query Processor**: `QueryProcessor` class handles natural language to pandas code conversion with security validation
- **Groq Client**: `GroqClient` class manages API interactions with the Groq service

### Security Architecture
- **Code Sanitization**: Whitelist-based approach allowing only safe pandas and numpy functions
- **Pattern Filtering**: Regular expression-based detection of dangerous code patterns
- **Sandboxed Execution**: Controlled execution environment for generated pandas code
- **Input Validation**: File type and format validation before processing

### Data Processing Pipeline
1. File upload and format detection
2. Automatic encoding detection for CSV files
3. DataFrame loading and validation
4. Natural language query processing
5. Code generation via Groq API
6. Security validation of generated code
7. Safe execution and result presentation

## External Dependencies

### AI/ML Services
- **Groq API**: Primary service for natural language to code conversion using Llama3-70b-8192 model
- **Model Configuration**: Low temperature (0.1) for consistent code generation, 1000 max tokens

### Python Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive data visualization (express and graph_objects)
- **numpy**: Numerical computing support
- **openpyxl**: Excel file reading capabilities
- **requests**: HTTP client for API communications

### Environment Configuration
- **GROQ_API_KEY**: Required environment variable for Groq API authentication
- **File Support**: CSV, XLSX, XLS file formats with automatic encoding detection

### Data Visualization
- **Plotly Express**: Simple statistical plots and charts
- **Plotly Graph Objects**: Advanced customizable visualizations
- **Streamlit Native**: Built-in data display components