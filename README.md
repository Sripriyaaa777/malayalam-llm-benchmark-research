# Malayalam-English LLM Benchmark v2 — 6 Models

**"Output Validity and Sentiment Accuracy of LLMs on Malayalam-English Code-Mixed Text"**

---

## Models evaluated

| Model | Provider | API | Size | Free? |
|---|---|---|---|---|
| Llama 3.3 70B | Meta | Groq | 70B | ✅ |
| Llama 4 Scout | Meta | Groq | 17B MoE (109B total) | ✅ |
| Gemma 2 9B | Google | Groq | 9B | ✅ (deprecated on Groq Aug 2025 — kept for paper reproduction) |
| Qwen 3 32B | Alibaba | Groq | 32B | ✅ |
| Mistral Large | Mistral AI | Mistral API | ~100B MoE | ✅ |
| Gemini 2.5 Flash | Google | Google AI Studio | — | ✅ |

**3 API keys, all free, no credit card required.**

---

## Setup

### 1. Install dependencies
```bash
cd malayalam_benchmark
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get your 3 API keys

| Key | Where to get it |
|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `MISTRAL_API_KEY` | https://console.mistral.ai/api-keys |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey |

All free. No credit card. Takes ~2 minutes total.

### 3. Configure .env
```bash
cp .env.template .env
# Open .env and paste your 3 keys
```

---

## Running

### Run everything at once (~90–120 min)
```bash
cd scripts/
python run_all.py
```

### Run individual experiments
```bash
cd scripts/

python exp1_2_zeroshot_3shot.py   # Exp 1 & 2: 0-shot / 3-shot, 100 samples  (~10 min)
python exp3_5shot_100.py          # Exp 3: 5-shot improved, 100 samples        (~8 min)
python exp4_5_500sample.py        # Exp 4 & 5: 500 samples, all 6 models       (~70 min)
python exp6_romanization.py       # Exp 6: Romanization control (needs Exp 4/5) (~20 min)
python statistical_tests.py       # Stats (needs Exp 4/5 + 6)
python error_analysis.py          # Error analysis (needs Exp 4/5)
python generate_metrics_matrix.py # Master table (run last)
```

### Resume after a crash
```bash
# Example: Exps 1–5 done, crashed during Exp 6
python run_all.py --skip-exp1-2 --skip-exp3 --skip-exp4-5
```

---

## Output files (all in results/)

| File | Contents |
|---|---|
| `exp4_5_500sample_*.csv` | Raw predictions — all 6 models, 500 samples |
| `exp6_romanization_*.csv` | Llama native vs romanised |
| `metrics_matrix_*.txt` | **Master paper table** — validity / cond.acc / e2e / macro-F1 |
| `per_class_tables_*.csv` | Per-class P / R / F1 |
| `prompting_progression_*.csv` | 0-shot → 3-shot → 5-shot |
| `statistical_tests_*.txt` | McNemar, chi-square, Cohen's h, bootstrap CIs |
| `error_analysis_*.csv` | Categorised misclassifications |
| `confusion_matrix_*.csv` | Confusion matrix (Mistral) |

---

## Rate limits (free tier)

| Provider | RPM | RPD | Notes |
|---|---|---|---|
| Groq | ~30 | 14,400 | Shared across all Groq models |
| Mistral | ~5–10 | ~1,000 | Mistral Large specifically |
| Google AI Studio | 15 | 1,500 | Gemini 2.5 Flash |

Scripts include automatic backoff — if you hit a rate limit, the script waits and retries automatically.

## Troubleshooting

**`aksharamukha` error**: Only needed for Exp 6. `pip install aksharamukha`

**Gemma 2 returns all INVALID**: Groq deprecated `gemma2-9b-it` in Aug 2025. This is the expected result and reproduces the original paper finding. The INVALID results are saved correctly.

**Gemini `google-generativeai` version error**: `pip install --upgrade google-generativeai`

**Mistral SDK version conflict**: Both old (<1.0) and new (≥1.0) SDKs are supported automatically.
