# 🤖 AI Data Agent - Scalable Data Analysis Platform

A production-ready, scalable Streamlit application that functions as an intelligent data agent, allowing users to upload tabular data and ask questions in natural language. Built with modern architectural patterns for seamless migration to React/backend systems.

## ✨ Features

### Core Capabilities
- **Natural Language Data Analysis**: Ask questions about your data in plain English
- **Multi-format Support**: CSV and Excel file uploads with comprehensive validation
- **AI-Powered Insights**: Powered by Groq's high-performance LLM API with pandas-ai
- **Interactive Visualizations**: Auto-generated charts with Plotly and Matplotlib
- **Query History**: Complete interaction tracking with JSONL storage
- **Real-time Processing**: Reactive UI with exponential backoff and retry logic

### Scalable Architecture
- **Modular Design**: Clean separation between UI, business logic, and data layers
- **LLM Abstraction**: Easy switching between different AI models (Groq, OpenAI, local models)
- **Query Processing Pipeline**: Pluggable architecture for custom processing logic
- **Visualization Engine**: Support for multiple charting libraries with unified interface
- **Storage Abstraction**: File-based storage ready for database migration
- **Comprehensive Logging**: Production-ready logging with multiple levels and handlers

### Enterprise Features
- **Error Handling**: Robust error management with user-friendly messages
- **Rate Limiting**: Intelligent API rate limiting with exponential backoff
- **Caching**: Data loading optimization with automatic cache management
- **Configuration Management**: Environment-based configuration for different deployment stages
- **Export Capabilities**: Chart and history export functionality
- **Session Management**: Streamlit session state optimization for scalability

## 🛠️ Tech Stack

### Core Technologies
- **Frontend**: Streamlit 1.33.0 (migration-ready for React)
- **Data Processing**: pandas 1.5.3 with comprehensive validation
- **AI/LLM**: pandas-ai 2.0.28 with Groq integration (langchain-groq 0.1.3)
- **Visualization**: Plotly 5.15.0 + Matplotlib 3.7.2
- **Configuration**: python-dotenv 1.0.1
- **File Processing**: openpyxl 3.1.2 for Excel support

### Architectural Patterns
- **Factory Pattern**: For component instantiation and configuration
- **Strategy Pattern**: For interchangeable LLM and visualization engines
- **Observer Pattern**: For logging and history tracking
- **Adapter Pattern**: For different data sources and formats
- **Repository Pattern**: For future database integration

## 🏗️ Project Architecture & Scalability

### Current Modular Structure
```
project-root/
├── app.py                          # Streamlit UI layer (easily replaceable)
├── core/                          # Business logic (backend-ready)
│   ├── data_loader.py            # Data processing with caching
│   ├── llm_manager.py            # LLM abstraction layer
│   └── query_processor.py       # Query processing pipeline
├── visualization/                 # Presentation layer
│   └── vis.py                    # Chart generation and rendering
├── storage/                      # Data persistence layer
│   └── history_manager.py       # History and session management
└── logs/                         # Application logging
```

### Migration Path to React/Backend

#### Phase 1: API Extraction (Current → REST API)
1. **Core modules** (`core/`) become FastAPI/Flask backend services
2. **Storage layer** migrates to PostgreSQL/MongoDB with existing interfaces
3. **Visualization** becomes API endpoints returning chart configurations
4. **Authentication** layer added using existing user management hooks

#### Phase 2: Frontend Migration (Streamlit → React)
1. **React frontend** consumes REST APIs from Phase 1
2. **Real-time features** via WebSocket connections
3. **State management** with Redux/Zustand using existing session patterns
4. **Component library** built around current visualization abstractions

#### Phase 3: Microservices Architecture
1. **LLM Service**: Dedicated service for AI processing
2. **Data Service**: File processing and validation
3. **Visualization Service**: Chart generation and export
4. **History Service**: User interactions and analytics
5. **API Gateway**: Request routing and rate limiting

### Abstraction Layers for Easy Component Switching

#### LLM Abstraction (`core/llm_manager.py`)
- **Current**: Groq integration with retry logic
- **Future**: OpenAI, Anthropic, local models (Ollama, Hugging Face)
- **Interface**: Unified response format and error handling
- **Configuration**: Environment-based model selection

#### Query Processing (`core/query_processor.py`)
- **Current**: pandas-ai integration
- **Future**: Custom NLP pipelines, LangChain agents, SQL generation
- **Interface**: Standardized QueryResult format
- **Plugins**: Extensible processor registration system

#### Visualization (`visualization/vis.py`)
- **Current**: Plotly + Matplotlib support
- **Future**: D3.js, Chart.js, custom React components
- **Interface**: Unified chart configuration and export
- **Engines**: Pluggable visualization backend system

#### Storage (`storage/history_manager.py`)
- **Current**: JSONL file-based storage
- **Future**: PostgreSQL, MongoDB, Redis caching
- **Interface**: Repository pattern for easy database switching
- **Migration**: Existing data structure compatible with relational/document stores

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ 
- Groq API key (free tier available)
- Git (for version control)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project-root
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Copy .env template
   cp .env.example .env
   # Edit .env with your Groq API key
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Quick Start with Sample Data
1. Open the application in your browser (typically `http://localhost:8501`)
2. Enter your Groq API key in the sidebar
3. Click "Load Sample Data" to use the included student dataset
4. Try example queries or ask your own questions

## 📖 How to Use

### Basic Workflow
1. **Setup**: Enter your Groq API key in the sidebar
2. **Data Upload**: Upload CSV/Excel files or use sample data
3. **Query**: Ask questions in natural language
4. **Analyze**: View results and auto-generated visualizations
5. **Export**: Save charts and export history as needed

### Example Queries
```text
Data Exploration:
- "What is the average stipend by department?"
- "Show me the distribution of students by gender"
- "Which department has the highest average stipend?"

Visualization Requests:
- "Create a bar chart of student count by department"
- "Show correlation between age and stipend"
- "Generate a pie chart of gender distribution"

Statistical Analysis:
- "What is the standard deviation of stipends?"
- "Show outliers in the age data"
- "Calculate the median stipend for each department"

Comparative Analysis:
- "Compare male vs female average stipends"
- "Which age group has the highest stipends?"
- "Show department performance metrics"
```

### Advanced Features
- **History Search**: Find previous queries and results
- **Export Options**: Download charts and data in multiple formats
- **Debug Mode**: View detailed processing information
- **Auto-visualization**: Toggle automatic chart generation
- **Session Management**: Maintain state across queries

## ⚙️ Configuration

### Environment Variables
```bash
# Required
GROQ_API_KEY="your-groq-api-key-here"

# Optional LLM Configuration
GROQ_MODEL="mixtral-8x7b-32768"          # Default model
GROQ_TEMPERATURE="0.1"                    # Response creativity (0-1)
GROQ_MAX_RETRIES="5"                     # API retry attempts
GROQ_TIMEOUT="30"                        # Request timeout (seconds)

# Application Configuration
LOG_LEVEL="INFO"                         # Logging level
CACHE_DIR="cache"                        # Data caching directory
HISTORY_FILE="query_history.jsonl"      # History storage file
MAX_HISTORY_ENTRIES="10000"             # Maximum history entries
```

### Application Settings
- **Data Loading**: Configurable caching, validation, and preprocessing
- **Visualization**: Chart themes, color schemes, and export formats
- **History Management**: Retention policies and backup strategies
- **Logging**: Multiple handlers with configurable levels and formats

## 👨‍💻 Development Guidelines

### Code Organization
- **Modular Architecture**: Each module has a single responsibility
- **Interface-driven**: Abstract base classes for extensibility
- **Factory Patterns**: Centralized object creation and configuration
- **Error Handling**: Comprehensive exception hierarchy with logging
- **Documentation**: Extensive docstrings and inline comments

### Adding New Features

#### New LLM Provider
```python
# 1. Create provider class in core/llm_manager.py
class NewLLMProvider(LLMProvider):
    def initialize(self) -> None:
        # Provider initialization
    
    def generate_response(self, prompt: str) -> str:
        # Response generation logic

# 2. Register in LLMFactory
def create_new_provider(config):
    return NewLLMProvider(config)
```

#### New Visualization Type
```python
# 1. Add chart type to visualization/vis.py
class ChartType(Enum):
    NEW_CHART = "new_chart"

# 2. Implement in visualizer
def _create_new_chart(self, data, config, **kwargs):
    # Chart creation logic
```

#### New Storage Backend
```python
# 1. Create repository in storage/
class DatabaseHistoryManager(HistoryManager):
    def _save_entry_to_file(self, entry):
        # Database save logic
```

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-module functionality
- **End-to-End Tests**: Complete user workflows
- **Performance Tests**: Load and stress testing
- **API Tests**: Backend endpoint validation (future)

### Code Quality
- **Type Hints**: Comprehensive type annotations
- **Linting**: Black, isort, pylint configuration
- **Documentation**: Sphinx-compatible docstrings
- **Error Handling**: Specific exception types with context
- **Logging**: Structured logging with correlation IDs

## 🗺️ Future Roadmap

### Phase 1: Enhanced AI Capabilities (Q1 2025)
- **Multi-model Support**: OpenAI, Anthropic, local model integration
- **Custom Prompting**: User-defined prompt templates and chains
- **Advanced Analytics**: Statistical testing and ML model integration
- **Real-time Processing**: Streaming data analysis capabilities

### Phase 2: React Frontend Migration (Q2 2025)
- **Modern UI**: React 18 with TypeScript
- **Component Library**: Reusable UI components with Storybook
- **State Management**: Redux Toolkit with RTK Query
- **Real-time Features**: WebSocket integration for live updates
- **Progressive Web App**: Offline capabilities and mobile optimization

### Phase 3: Backend API Development (Q3 2025)
- **FastAPI Backend**: High-performance async API
- **Database Integration**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: JWT-based auth with role-based access
- **API Documentation**: OpenAPI/Swagger with interactive docs
- **Containerization**: Docker and Kubernetes deployment

### Phase 4: Enterprise Features (Q4 2025)
- **Multi-tenancy**: Organization and team management
- **Advanced Security**: Data encryption and audit logging
- **Collaboration**: Shared workspaces and query templates
- **Integration APIs**: Third-party data source connectors
- **Analytics Dashboard**: Usage metrics and performance monitoring

### Phase 5: Advanced Analytics Platform (2026)
- **Machine Learning**: AutoML pipeline integration
- **Custom Dashboards**: Drag-and-drop dashboard builder
- **Alerting System**: Data anomaly detection and notifications
- **Data Governance**: Lineage tracking and compliance features
- **Marketplace**: Community-contributed analysis templates

### Deployment Strategies
- **Development**: Local development with hot reload
- **Staging**: Docker Compose with integrated testing
- **Production**: Kubernetes with auto-scaling and monitoring
- **Cloud**: AWS/Azure/GCP deployment templates
- **Edge**: CDN integration for global performance

## 🔧 Troubleshooting

### Common Issues

#### API Key Problems
```
Error: "Groq API key not provided or still using placeholder"
Solution: Update .env file with valid Groq API key
```

#### Import Errors
```
Error: "Import 'pandasai' could not be resolved"
Solution: pip install -r requirements.txt
```

#### File Upload Issues
```
Error: "Data validation failed"
Solution: Check file format (CSV/Excel) and data structure
```

#### Memory Issues
```
Error: "DataFrame too large"
Solution: Increase system memory or use data sampling
```

### Performance Optimization
- **Data Caching**: Enable caching for repeated file loads
- **Query Optimization**: Use specific column references in queries
- **Visualization**: Limit chart complexity for large datasets
- **History Management**: Regular cleanup of old entries
- **API Rate Limiting**: Implement request queuing for high usage

### Debug Mode
Enable debug mode in the sidebar to access:
- **Request/Response Logs**: Detailed API interaction logs
- **Performance Metrics**: Query execution times and resource usage
- **State Inspection**: Session state and component status
- **Error Details**: Full stack traces and context information

## 📊 Example Queries for Testing

### Data Exploration
```text
"What does this dataset contain?"
"Show me the first 10 rows"
"How many rows and columns are there?"
"What are the data types of each column?"
"Are there any missing values?"
```

### Statistical Analysis
```text
"Calculate summary statistics for all numeric columns"
"What is the correlation between age and stipend?"
"Show the distribution of students by department"
"Find outliers in the stipend data"
"What is the average, median, and mode of ages?"
```

### Visualization Requests
```text
"Create a bar chart showing student count by department"
"Generate a scatter plot of age vs stipend"
"Show a pie chart of gender distribution"
"Create a histogram of stipend amounts"
"Display a box plot of stipends by department"
```

### Comparative Analysis
```text
"Compare average stipends between male and female students"
"Which department has the most students?"
"Show the highest and lowest stipends by department"
"What percentage of students are in each department?"
"Compare age distributions across departments"
```

### Advanced Queries
```text
"Find students with stipends above the 75th percentile"
"Show departments with above-average stipend variance"
"Create a correlation matrix for all numeric variables"
"Identify the top 3 departments by average stipend"
"Calculate the gender ratio in each department"
```

## 🤝 Contributing

We welcome contributions to make this platform even better! Please read our contributing guidelines and follow the established code patterns.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Follow the existing code style and patterns
4. Add tests for new functionality
5. Update documentation as needed
6. Submit a pull request with detailed description

### Areas for Contribution
- New LLM provider integrations
- Additional visualization types
- Performance optimizations
- Documentation improvements
- Testing coverage expansion
- UI/UX enhancements

---

**Built with ❤️ for scalable data analysis | Ready for enterprise deployment and React migration**
