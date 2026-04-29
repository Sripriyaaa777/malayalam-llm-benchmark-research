"""
test_api_keys.py — Sends ONE test call to each API and prints the raw response.
Run this BEFORE the full experiment to confirm keys work and see what format
each model returns. Takes ~30 seconds total.

Usage:
    python test_api_keys.py
    python test_api_keys.py --skip-gemini
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

TEST_TEXT = "Super movie! BGM kollaam, climax scene vere level aayirunnu!"
EXPECTED  = "Positive"

def test_groq_model(client, model_id, label):
    from api_clients import SYSTEM_5SHOT, clean_prediction
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_5SHOT},
                {"role": "user",   "content": f'Text: "{TEST_TEXT}"\nSentiment:'},
            ],
            max_tokens=20, temperature=0,
        )
        raw = resp.choices[0].message.content
        parsed = clean_prediction(raw.strip())
        status = "✓ PASS" if parsed == EXPECTED else f"⚠ PARSED={parsed}"
        print(f"  [{status}] {label}")
        print(f"         raw = {repr(raw)}")
    except Exception as e:
        print(f"  [✗ FAIL] {label}")
        print(f"         error = {str(e)[:120]}")

def test_mistral(client, sdk_ver):
    from api_clients import SYSTEM_5SHOT, clean_prediction
    messages = [
        {"role":"system","content":SYSTEM_5SHOT},
        {"role":"user","content":f'Text: "{TEST_TEXT}"\nSentiment:'},
    ]
    try:
        if sdk_ver == "new":
            resp = client.chat.complete(model="mistral-large-latest", messages=messages,
                                        max_tokens=20, temperature=0)
            raw = resp.choices[0].message.content
        else:
            from mistralai.models.chat_completion import ChatMessage
            cms = [ChatMessage(role=m["role"],content=m["content"]) for m in messages]
            resp = client.chat(model="mistral-large-latest", messages=cms,
                               max_tokens=20, temperature=0)
            raw = resp.choices[0].message.content
        parsed = clean_prediction(raw.strip())
        status = "✓ PASS" if parsed == EXPECTED else f"⚠ PARSED={parsed}"
        print(f"  [{status}] Mistral Large")
        print(f"         raw = {repr(raw)}")
    except Exception as e:
        print(f"  [✗ FAIL] Mistral Large")
        print(f"         error = {str(e)[:120]}")

def test_gemini(client):
    from api_clients import SYSTEM_5SHOT, clean_prediction
    try:
        from google.genai import types
        full = f"{SYSTEM_5SHOT}\n\nText: \"{TEST_TEXT}\"\nSentiment:"
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full,
            config=types.GenerateContentConfig(
                max_output_tokens=20,
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = resp.text
        parsed = clean_prediction(raw.strip())
        status = "✓ PASS" if parsed == EXPECTED else f"⚠ PARSED={parsed}"
        print(f"  [{status}] Gemini 2.5 Flash")
        print(f"         raw = {repr(raw)}")
    except Exception as e:
        print(f"  [✗ FAIL] Gemini 2.5 Flash")
        print(f"         error = {str(e)[:120]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gemini", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("API KEY + FORMAT TEST")
    print(f"Test input:  {repr(TEST_TEXT)}")
    print(f"Expected:    {EXPECTED}")
    print("="*60)

    from api_clients import _get_groq_client, _get_mistral_client, _get_gemini_client

    # Groq models
    print("\n--- GROQ ---")
    try:
        groq = _get_groq_client()
        for model_id, label in [
            ("llama-3.3-70b-versatile",                    "Llama 3.3 70B"),
            ("meta-llama/llama-4-scout-17b-16e-instruct",  "Llama 4 Scout"),
            ("gemma2-9b-it",                               "Gemma 2 9B"),
            ("qwen/qwen3-32b",                             "Qwen 3 32B"),
        ]:
            test_groq_model(groq, model_id, label)
    except Exception as e:
        print(f"  [✗] Could not create Groq client: {e}")

    # Mistral
    print("\n--- MISTRAL ---")
    try:
        mc, sdk = _get_mistral_client()
        test_mistral(mc, sdk)
    except Exception as e:
        print(f"  [✗] Could not create Mistral client: {e}")

    # Gemini
    if not args.skip_gemini:
        print("\n--- GEMINI ---")
        try:
            gem = _get_gemini_client()
            test_gemini(gem)
        except Exception as e:
            print(f"  [✗] Could not create Gemini client: {e}")

    print("\n" + "="*60)
    print("If any model shows ⚠ PARSED=INVALID, run:")
    print("  python diagnose_results.py  PATH_TO_YOUR_CSV")
    print("to see the raw outputs and fix clean_prediction() accordingly.")

if __name__ == "__main__":
    main()
