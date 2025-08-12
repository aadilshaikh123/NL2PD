# PowerShell startup script for NL2PD application
# This script handles SSL certificate issues and starts the Streamlit app

Write-Host "Starting NL2PD Application..." -ForegroundColor Green
Write-Host ""

# Activate virtual environment if not already activated
if (-not $env:VIRTUAL_ENV) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "nlp\Scripts\Activate.ps1"
}

# Clear problematic SSL environment variables
Write-Host "Configuring SSL environment..." -ForegroundColor Yellow

# Remove SSL environment variables at multiple levels
Remove-Item Env:SSL_CERT_FILE -ErrorAction SilentlyContinue
Remove-Item Env:REQUESTS_CA_BUNDLE -ErrorAction SilentlyContinue
Remove-Item Env:CURL_CA_BUNDLE -ErrorAction SilentlyContinue

# Force clear using different methods
$env:SSL_CERT_FILE = $null
$env:REQUESTS_CA_BUNDLE = $null
$env:CURL_CA_BUNDLE = $null

# Also try to unset them completely
[Environment]::SetEnvironmentVariable("SSL_CERT_FILE", $null, "Process")
[Environment]::SetEnvironmentVariable("REQUESTS_CA_BUNDLE", $null, "Process")
[Environment]::SetEnvironmentVariable("CURL_CA_BUNDLE", $null, "Process")

# Set proper certificate bundle path using certifi
try {
    $certPath = & python -c "import certifi; print(certifi.where())"
    $env:REQUESTS_CA_BUNDLE = $certPath
    Write-Host "SSL certificate bundle configured: $certPath" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not configure SSL certificate bundle" -ForegroundColor Yellow
}

# Load environment variables from .env file
try {
    $envContent = Get-Content ".env" -ErrorAction SilentlyContinue
    foreach ($line in $envContent) {
        if ($line -match '^([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"')
            Set-Item -Path "env:$name" -Value $value
            Write-Host "Loaded environment variable: $name" -ForegroundColor Green
        }
    }
} catch {
    Write-Host "Warning: Could not load .env file" -ForegroundColor Yellow
}

Write-Host ""

# Check if GROQ_API_KEY is set
if (-not $env:GROQ_API_KEY) {
    Write-Host "WARNING: GROQ_API_KEY environment variable is not set!" -ForegroundColor Red
    Write-Host "Please set your Groq API key before running the application." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You can set it by running:" -ForegroundColor Yellow
    Write-Host '$env:GROQ_API_KEY = "your_api_key_here"' -ForegroundColor Cyan
    Write-Host ""
    Read-Host "Press Enter to continue anyway, or Ctrl+C to exit"
} else {
    Write-Host "✅ GROQ_API_KEY found" -ForegroundColor Green
}

# Start the Streamlit application
Write-Host ""
Write-Host "Starting Streamlit application..." -ForegroundColor Green
try {
    streamlit run app.py
} catch {
    Write-Host "Error starting Streamlit: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
