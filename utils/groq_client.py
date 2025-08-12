import os
import requests
import json
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import urllib3
import certifi

# Load environment variables from .env file
load_dotenv()


class GroqClient:
    """Client for interacting with Groq API using Llama3-70b-8192 model"""
    
    _ssl_configured = False  # Class variable to track SSL configuration
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192"
        self.max_tokens = 1000
        self.temperature = 0.1  # Low temperature for more consistent code generation
        
        # Handle SSL certificate issues only once
        if not GroqClient._ssl_configured:
            self._setup_ssl_handling()
            GroqClient._ssl_configured = True
        
        # Create a custom session with SSL configuration
        self.session = self._create_ssl_session()
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
    
    def _create_ssl_session(self):
        """Create a requests session with proper SSL configuration"""
        try:
            session = requests.Session()
            
            # Set the certificate bundle
            session.verify = certifi.where()
            
            # Configure SSL adapter with retry logic
            adapter = requests.adapters.HTTPAdapter(
                max_retries=requests.packages.urllib3.util.retry.Retry(
                    total=2,
                    backoff_factor=0.3,
                    status_forcelist=[500, 502, 504]
                )
            )
            session.mount('https://', adapter)
            
            return session
            
        except Exception as e:
            print(f"Warning: Could not create SSL session: {e}")
            # Fallback to basic session
            session = requests.Session()
            session.verify = False  # Disable SSL verification as last resort
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            return session
    
    def _setup_ssl_handling(self):
        """Setup SSL handling to avoid certificate issues"""
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Aggressively fix SSL environment variables
        ssl_cert_file = os.environ.get('SSL_CERT_FILE', '')
        
        # Always remove problematic SSL_CERT_FILE if it points to PostgreSQL
        if ssl_cert_file and ('PostgreSQL' in ssl_cert_file or not os.path.exists(ssl_cert_file)):
            print(f"Removing problematic SSL_CERT_FILE: {ssl_cert_file}")
            if 'SSL_CERT_FILE' in os.environ:
                del os.environ['SSL_CERT_FILE']
        
        # Force set the certificate bundle to certifi's path
        certifi_path = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi_path
        os.environ['SSL_CERT_FILE'] = certifi_path
        
        print(f"SSL certificate bundle set to: {certifi_path}")
        
        # Also clear other potentially problematic variables
        for var in ['CURL_CA_BUNDLE', 'SSL_CERT_DIR']:
            if var in os.environ:
                del os.environ[var]
    
    def generate_code(self, prompt: str) -> str:
        """
        Generate pandas code using Groq Llama3-70b-8192 model
        
        Args:
            prompt: The prompt containing DataFrame info and user query
            
        Returns:
            Generated pandas code as string
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a pandas expert code generator. Generate only clean, "
                        "executable pandas code without any explanations, comments, or "
                        "markdown formatting. Return only the code that directly answers "
                        "the user's query using the provided DataFrame 'df'."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": 1,
                "stop": None
            }
            
            # Use the custom session for the request with error handling
            try:
                response = self.session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
            except Exception as session_error:
                # If session fails, try with basic requests as fallback
                print(f"Session request failed, trying fallback: {session_error}")
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=30,
                    verify=certifi.where()
                )
            
            if response.status_code == 200:
                response_data = response.json()
                return self._extract_code_from_response(response_data)
            else:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Request to Groq API timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error generating code with Groq: {str(e)}")
    
    def _extract_code_from_response(self, response_data: Dict[str, Any]) -> str:
        """Extract code from Groq API response"""
        try:
            if 'choices' not in response_data or not response_data['choices']:
                raise Exception("No choices in response")
            
            content = response_data['choices'][0]['message']['content']
            
            if not content:
                raise Exception("Empty response content")
            
            # Clean the response
            code = content.strip()
            
            # Remove any markdown formatting
            if code.startswith('```python'):
                code = code[9:]
            elif code.startswith('```'):
                code = code[3:]
            
            if code.endswith('```'):
                code = code[:-3]
            
            # Remove any explanatory text before or after code
            lines = code.split('\n')
            code_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip comments and explanations
                if line and not line.startswith('#') and not line.startswith('//'):
                    # Check if line looks like pandas/python code
                    if any(keyword in line for keyword in ['df', 'pd.', 'np.', '=', '(', ')', '[', ']']):
                        code_lines.append(line)
                    elif line and not any(word in line.lower() for word in ['here', 'this', 'will', 'the', 'code', 'query', 'result']):
                        code_lines.append(line)
            
            if not code_lines:
                # Fallback: return the cleaned code as-is
                return code.strip()
            
            return '\n'.join(code_lines)
            
        except Exception as e:
            raise Exception(f"Error extracting code from response: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Groq API"""
        try:
            test_prompt = """
DataFrame Information:
- Shape: 5 rows × 2 columns
- Columns: ['name', 'age']

User Query: "show first row"

Generate ONLY the pandas code to answer this query. The DataFrame is already loaded as 'df'.
Your code:
"""
            
            result = self.generate_code(test_prompt)
            
            return {
                'success': True,
                'message': 'Connection successful',
                'test_result': result
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}',
                'test_result': None
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'api_endpoint': self.base_url
        }
