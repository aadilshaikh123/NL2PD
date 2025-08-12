@echo off
REM Startup script for NL2PD application
REM This script handles SSL certificate issues and starts the Streamlit app

echo Starting NL2PD Application...
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call nlp\Scripts\activate.bat
)

REM Clear problematic SSL environment variables
set SSL_CERT_FILE=
set REQUESTS_CA_BUNDLE=
set CURL_CA_BUNDLE=

REM Set proper certificate bundle path
for /f "tokens=*" %%i in ('python -c "import certifi; print(certifi.where())"') do set REQUESTS_CA_BUNDLE=%%i

echo SSL environment configured...
echo.

REM Check if GROQ_API_KEY is set
if not defined GROQ_API_KEY (
    echo WARNING: GROQ_API_KEY environment variable is not set!
    echo Please set your Groq API key before running the application.
    echo.
    echo You can set it by running:
    echo set GROQ_API_KEY=your_api_key_here
    echo.
    pause
)

REM Start the Streamlit application
echo Starting Streamlit application...
streamlit run app.py

pause
