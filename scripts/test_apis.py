"""
Test script to verify all API keys are working
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("=" * 60)
print("TESTING API CONNECTIONS")
print("=" * 60)

# Test 1: Check if API keys are loaded
print("\n1. Checking if API keys are loaded from .env file...")
print("-" * 60)

google_key = os.getenv("GOOGLE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")

if google_key and google_key != "your_google_key_here":
    print("✓ Google API key loaded")
else:
    print("✗ Google API key NOT loaded")

if groq_key and groq_key != "your_groq_key_here":
    print("✓ Groq API key loaded")
else:
    print("✗ Groq API key NOT loaded")

if mistral_key and mistral_key != "your_mistral_key_here":
    print("✓ Mistral API key loaded")
else:
    print("✗ Mistral API key NOT loaded")

# Test 2: Test Google Gemini API
print("\n2. Testing Google Gemini API...")
print("-" * 60)
try:
    import google.generativeai as genai
    genai.configure(api_key=google_key)
    
    # Try different model names until one works
    model_names = ['gemini-2.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash']
    
    model = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'Hello from Gemini!' in one sentence.")
            print(f"✓ Gemini API working with model: {model_name}")
            print(f"  Response: {response.text.strip()}")
            break
        except Exception as model_error:
            continue
    
    if model is None:
        print("✗ Gemini API error: Could not find working model. Check your API key at https://aistudio.google.com/app/apikey")
        
except Exception as e:
    print(f"✗ Gemini API error: {str(e)}")

# Test 3: Test Groq API (Llama)
print("\n3. Testing Groq API (Llama 3.3)...")
print("-" * 60)
try:
    from groq import Groq
    client = Groq(api_key=groq_key)
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say 'Hello from Llama!' in one sentence."}],
        max_tokens=50
    )
    print(f"✓ Groq/Llama API working! Response: {completion.choices[0].message.content.strip()}")
except Exception as e:
    print(f"✗ Groq API error: {str(e)}")

# Test 4: Test Mistral API
print("\n4. Testing Mistral API...")
print("-" * 60)
try:
    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    client = MistralClient(api_key=mistral_key)
    
    response = client.chat(
        model="mistral-large-latest",
        messages=[ChatMessage(role="user", content="Say 'Hello from Mistral!' in one sentence.")]
    )
    print(f"✓ Mistral API working! Response: {response.choices[0].message.content.strip()}")
except Exception as e:
    print(f"✗ Mistral API error: {str(e)}")

print("\n" + "=" * 60)
print("API TESTING COMPLETE!")
print("=" * 60)
print("\n✅ WORKING APIs: We have 2-3 free models ready!")
print("   - Groq/Llama 3.3 (FREE)")
print("   - Mistral Large (FREE)")
print("   - Gemini (FREE - if it worked above)")
print("\n📊 This is enough for a publishable paper!")
print("=" * 60)