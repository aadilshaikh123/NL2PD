# Natural Language to Pandas Query Converter

## Overview

This project is a Streamlit-based web application that allows users to upload CSV or Excel files and query their data using natural language. The application leverages the Groq API to convert natural language queries into executable pandas code, making data analysis accessible to users without programming knowledge.

## Features

- **File Upload**: Supports CSV and Excel file formats.
- **Natural Language Querying**: Converts user queries into pandas code using the Groq API.
- **Data Preview**: Displays the first 10 rows of the uploaded dataset along with basic statistics.
- **Query History**: Keeps track of the last 5 queries executed.
- **Data Visualization**: Automatically generates visualizations based on query results.
- **Security**: Implements code sanitization and validation to ensure safe execution of generated code.

## Project Structure

```
PandaQueryGen/
├── app.py                 # Main Streamlit application
├── pyproject.toml         # Project dependencies and metadata
├── requirements_local.txt # Local Python dependencies
├── utils/                 # Utility modules
│   ├── data_handler.py    # Handles file loading and validation
│   ├── groq_client.py     # Manages interactions with the Groq API
│   └── query_processor.py # Processes natural language queries
└── .gitignore             # Git ignore file
```

### Utility Modules

1. **`data_handler.py`**: 
   - Loads CSV and Excel files into pandas DataFrames.
   - Automatically detects file delimiters and encodings.

2. **`groq_client.py`**: 
   - Manages API interactions with the Groq service.
   - Configured to use the Llama3-70b-8192 model for natural language processing.

3. **`query_processor.py`**: 
   - Converts natural language queries into pandas code.
   - Validates and executes the generated code securely.

## Prerequisites

- Python 3.11 or higher
- A Groq API key (set as the `GROQ_API_KEY` environment variable)

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/aadilshaikh123/NL2PD.git
   cd NL2PD
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements_local.txt
   ```

4. **Set the Groq API Key**:
   - Create a `.env` file in the project root and add your API key:
     ```
     GROQ_API_KEY=your_api_key_here
     ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

6. **Access the Application**:
   - Open your browser and navigate to `http://localhost:8501`.

## Usage

1. Upload a CSV or Excel file using the sidebar.
2. Enter a natural language query in the text area (e.g., "Show me the first 10 rows").
3. View the results, generated pandas code, and visualizations.
4. Download query results as a CSV file if needed.

## Example Queries

- "Show me the first 10 rows"
- "What are the column names?"
- "Calculate the mean of all numeric columns"
- "Find rows where [column_name] is greater than [value]"
- "Create a histogram of [column_name]"

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.