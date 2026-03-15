import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY")

print("Testing Gemini API in detail...")
print(f"API Key loaded: {google_key[:10]}... (showing first 10 chars)")
print("-" * 60)

genai.configure(api_key=google_key)

# List available models
print("\nAvailable Gemini models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")

# Try to use a model
print("\nTrying to generate content...")
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content("Say hello")
    print(f"✓ Success! Response: {response.text}")
except Exception as e:
    print(f"✗ Error: {e}")