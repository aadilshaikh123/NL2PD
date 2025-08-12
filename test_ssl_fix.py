#!/usr/bin/env python3
"""
Test script to verify SSL fix for Groq client
"""
import os
import sys
sys.path.append('.')

from utils.groq_client import GroqClient

def test_ssl_fix():
    """Test if SSL fix resolves the certificate issue"""
    print("Testing SSL fix for Groq client...")
    
    # Check if GROQ_API_KEY is available
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment variables")
        print("Please set your Groq API key to test the connection")
        return False
    
    try:
        # Initialize Groq client (this will apply SSL fixes)
        client = GroqClient()
        print("✅ Groq client initialized successfully")
        
        # Test connection
        result = client.test_connection()
        
        if result['success']:
            print("✅ Connection test passed")
            print(f"Test result: {result['test_result']}")
            return True
        else:
            print(f"❌ Connection test failed: {result['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_ssl_fix()
    print("\n" + "="*50)
    if success:
        print("🎉 SSL fix is working! Your app should now work properly.")
    else:
        print("⚠️  SSL fix test failed. Check your API key and connection.")
    print("="*50)
