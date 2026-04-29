"""
Centralised LLM API clients with retry + rate-limit backoff.

Models (all free-tier):
  GROQ key:    Llama 3.3 70B | Llama 4 Scout | Gemma 3 12B | Qwen 3 32B (thinking disabled)
  MISTRAL key: Mistral Large
  GEMINI key:  Gemini 2.5 Flash (google-genai SDK, thinking disabled)
"""
import os, sys, re, time
from dotenv import load_dotenv
load_dotenv()

VALID_LABELS = ['Positive', 'Negative', 'Mixed_feelings']

FIVE_SHOT_BLOCK = """Here are five examples of Malayalam-English code-mixed sentiment classification:

Example 1:
Text: "ഈ പടം കിടു ആണ്! Climax scene വേറെ level! Totally paisa vasool. Must watch!"
Sentiment: Positive

Example 2:
Text: "Bore adichu mari. First half okay aarunnu but second half വല്യ waste. Time and money പോയി."
Sentiment: Negative

Example 3:
Text: "Songs കൊള്ളാം, bgm നന്നായി. But story weak aanu. Average padam എന്ന് പറയാം."
Sentiment: Mixed_feelings

Example 4:
Text: "Adipoli performance! Hero mass aanu. Interval scene kollaam. Theatre il energy vere level!"
Sentiment: Positive

Example 5:
Text: "Trailer kandappo excited aayi but padam disappointment aayi. Expected ഒന്നും illatha feel."
Sentiment: Negative
"""

SYSTEM_5SHOT = (
    "You are a sentiment classifier for Malayalam-English code-mixed text (Manglish). "
    "Classify the sentiment as exactly one of: Positive, Negative, Mixed_feelings. "
    "Output ONLY the label word and nothing else.\n\n" + FIVE_SHOT_BLOCK
)
SYSTEM_0SHOT = (
    "You are a sentiment classifier for Malayalam-English code-mixed text (Manglish). "
    "Classify the sentiment as exactly one of: Positive, Negative, Mixed_feelings. "
    "Output ONLY the label word and nothing else."
)

def make_prompt(text, shot="5shot"):
    system = SYSTEM_5SHOT if shot == "5shot" else SYSTEM_0SHOT
    return system, f'Text: "{text}"\nSentiment:'

def strip_think_tags(raw):
    """Remove <think>...</think> blocks that Qwen 3 and some models emit."""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    return raw.strip()

def clean_prediction(raw):
    """
    Robust label parser handling all known API output variants:
      - exact / lowercase / with spaces or newlines
      - with surrounding quotes
      - "Mixed feelings" (no underscore)
      - prefixed: "Answer: Positive"
      - after <think> blocks (Qwen 3)
    """
    if not raw:
        return "INVALID"

    # Strip <think>...</think> first (Qwen 3 emits these)
    raw = strip_think_tags(raw)

    if not raw:
        return "INVALID"

    # Strip whitespace and surrounding quotes
    r = raw.strip().strip('\'"').strip()
    r_lower = r.lower()

    # 1. Exact lowercase match
    if r_lower in ("positive",):                          return "Positive"
    if r_lower in ("negative",):                          return "Negative"
    if r_lower in ("mixed_feelings", "mixed feelings",
                   "mixed-feelings", "mixedfeelings"):    return "Mixed_feelings"

    # 2. Substring scan (handles "Answer: Positive", sentence responses, etc.)
    for label in VALID_LABELS:
        if label in r:        return label
        if label.lower() in r_lower: return label

    # 3. "Mixed feelings" variant without underscore
    if "mixed" in r_lower and "feel" in r_lower:
        return "Mixed_feelings"

    # 4. Starts-with fallback
    if r_lower.startswith("positive"):  return "Positive"
    if r_lower.startswith("negative"):  return "Negative"
    if r_lower.startswith("mixed"):     return "Mixed_feelings"

    return "INVALID"


# ── Groq ──────────────────────────────────────────────────────────────────────
def _get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key: raise ValueError("GROQ_API_KEY not set in .env")
    try:
        from groq import Groq
        return Groq(api_key=key)
    except ImportError:
        raise ImportError("Run: pip install groq")

def _call_groq(client, model_id, text, shot="5shot",
               retries=3, extra_params=None):
    system, user = make_prompt(text, shot)
    params = dict(
        model=model_id,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        max_tokens=20, temperature=0,
    )
    if extra_params:
        params.update(extra_params)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(**params)
            return clean_prediction(resp.choices[0].message.content.strip())
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err or "too many" in err.lower():
                wait = 60*(attempt+1)
                print(f"\n  [Groq rate limit on {model_id}] waiting {wait}s…",
                      end="", flush=True)
                time.sleep(wait)
            elif any(x in err.lower() for x in
                     ["not found","deprecated","decommission","removed"]):
                return "INVALID"   # model gone — don't retry
            else:
                time.sleep(2)
    return "INVALID"

def predict_llama33(client, text, shot="5shot"):
    """Llama 3.3 70B — confirmed working."""
    return _call_groq(client, "llama-3.3-70b-versatile", text, shot)

def predict_llama4(client, text, shot="5shot"):
    """Llama 4 Scout 17B MoE — confirmed working."""
    return _call_groq(client, "meta-llama/llama-4-scout-17b-16e-instruct", text, shot)

def predict_gemma(client, text, shot="5shot"):
    """
    Gemma 3 12B via Groq.
    (Gemma 2 9B was decommissioned by Groq — replaced with Gemma 3 12B.)
    Paper note: original experiment used gemma2-9b-it which produced 0% validity.
    This new run uses gemma-3-12b-it as the updated Gemma baseline.
    """
    return _call_groq(client, "gemma-3-12b-it", text, shot)

def predict_qwen3(client, text, shot="5shot"):
    """
    Qwen 3 32B via Groq.
    Uses /no_think suffix in system prompt to suppress <think> blocks,
    AND strips any residual <think>...</think> tags in clean_prediction().
    """
    system, user = make_prompt(text, shot)
    # Qwen 3 respects /no_think appended to the system prompt
    system_no_think = system + "\n/no_think"
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[{"role":"system","content":system_no_think},
                          {"role":"user","content":user}],
                max_tokens=20, temperature=0,
            )
            return clean_prediction(resp.choices[0].message.content.strip())
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                time.sleep(60*(attempt+1))
            else:
                time.sleep(2)
    return "INVALID"


# ── Mistral ───────────────────────────────────────────────────────────────────
def _get_mistral_client():
    key = os.getenv("MISTRAL_API_KEY")
    if not key: raise ValueError("MISTRAL_API_KEY not set in .env")
    try:
        from mistralai import Mistral
        return Mistral(api_key=key), "new"
    except ImportError:
        pass
    try:
        from mistralai.client import MistralClient
        return MistralClient(api_key=key), "old"
    except ImportError:
        raise ImportError("Run: pip install mistralai")

def predict_mistral(client, sdk_ver, text, shot="5shot", retries=3):
    system, user = make_prompt(text, shot)
    messages = [{"role":"system","content":system},
                {"role":"user","content":user}]
    for attempt in range(retries):
        try:
            if sdk_ver == "new":
                resp = client.chat.complete(
                    model="mistral-large-latest",
                    messages=messages, max_tokens=20, temperature=0)
                return clean_prediction(resp.choices[0].message.content.strip())
            else:
                from mistralai.models.chat_completion import ChatMessage
                cms = [ChatMessage(role=m["role"],content=m["content"]) for m in messages]
                resp = client.chat(model="mistral-large-latest", messages=cms,
                                   max_tokens=20, temperature=0)
                return clean_prediction(resp.choices[0].message.content.strip())
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err:
                wait = 60*(attempt+1)
                print(f"\n  [Mistral rate limit] waiting {wait}s…", end="", flush=True)
                time.sleep(wait)
            else:
                time.sleep(2)
    return "INVALID"


# ── Gemini 2.5 Flash (google-genai SDK) ──────────────────────────────────────
def _get_gemini_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key: raise ValueError("GEMINI_API_KEY not set in .env")
    try:
        from google import genai
        return genai.Client(api_key=key)
    except ImportError:
        raise ImportError(
            "Run: pip install google-genai\n"
            "(NOT 'google-generativeai' — that package is deprecated)"
        )

def predict_gemini(client, text, shot="5shot", retries=3):
    """Gemini 2.5 Flash with thinking disabled for fast classification."""
    system, user = make_prompt(text, shot)
    full_prompt = f"{system}\n\n{user}"
    for attempt in range(retries):
        try:
            from google.genai import types
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=20,
                    temperature=0,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return clean_prediction(resp.text.strip())
        except Exception as e:
            err = str(e)
            if "quota" in err.lower() or "429" in err or "rate" in err.lower():
                wait = 60*(attempt+1)
                print(f"\n  [Gemini rate limit] waiting {wait}s…", end="", flush=True)
                time.sleep(wait)
            else:
                if attempt == retries-1:
                    print(f"\n  [Gemini error] {err[:100]}", end="", flush=True)
                time.sleep(2)
    return "INVALID"