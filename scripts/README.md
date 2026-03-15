\# Malayalam-English Code-Mixing LLM Benchmark



First systematic evaluation of Large Language Models on Malayalam-English code-mixed sentiment analysis.



\## 🎯 Project Overview



This project benchmarks modern LLMs (Llama 3.3 70B, Mistral Large) on Malayalam-English code-mixed text using few-shot prompting strategies.



\### Key Findings



1\. \*\*Few-shot prompting\*\* achieves 63.5% accuracy (\~23% improvement over zero-shot)

2\. \*\*Script-handling discovery\*\*: Llama 3.3 70B fails on 56% of samples with Malayalam script, while Mistral Large succeeds on 99.6%

3\. \*\*Mixed sentiment\*\* remains challenging (31% accuracy vs 70-75% for clear sentiment)



\## 📊 Results Summary



| Model | 100 Samples | 500 Samples | Script Handling |

|-------|-------------|-------------|-----------------|

| Llama 3.3 70B | 81.2% | N/A (56% failures) | Poor (44% success) |

| Mistral Large | 71.0% | 63.5% | Excellent (99.6%) |



\## 🚀 Quick Start



\### Prerequisites

\- Python 3.10+

\- Free API keys from:

&#x20; - \[Groq](https://console.groq.com/) (Llama)

&#x20; - \[Mistral AI](https://console.mistral.ai/)



\### Installation

```bash

\# Clone repository

git clone https://github.com/YOUR\_USERNAME/malayalam-llm-benchmark.git

cd malayalam-llm-benchmark



\# Create virtual environment

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate



\# Install dependencies

pip install -r requirements.txt



\# Download dataset

\# Follow instructions in data/README.md

```



\### Setup API Keys



Create `.env` file in project root:

```bash

GROQ\_API\_KEY=your\_groq\_key\_here

MISTRAL\_API\_KEY=your\_mistral\_key\_here

```



\### Run Experiments

```bash

\# Test APIs

python scripts/test\_apis.py



\# Run sentiment analysis (100 samples)

python scripts/sentiment\_5shot\_improved.py



\# Run large-scale experiment (500 samples)

python scripts/fixed\_large\_scale.py

```



\## 📁 Project Structure

```

malayalam-llm-benchmark/

├── data/                  # Dataset (download separately)

├── scripts/               # All Python scripts

│   ├── load\_data.py

│   ├── test\_apis.py

│   ├── sentiment\_5shot\_improved.py

│   └── fixed\_large\_scale.py

├── results/               # Experiment results (sample only)

├── .env                   # API keys (DO NOT COMMIT!)

├── .gitignore

├── requirements.txt

└── README.md

```



\## 📚 Dataset



We use the DravidianCodeMix dataset from FIRE 2020:

\- \*\*Source\*\*: \[DravidianCodeMix-Dataset](https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset)

\- \*\*Size\*\*: 9,323 Malayalam-English code-mixed samples

\- \*\*Task\*\*: Sentiment analysis (Positive, Negative, Mixed\_feelings)



Download instructions in `data/README.md`



\## 🔬 Novel Contributions



1\. \*\*First LLM benchmark\*\* for Malayalam-English code-mixing

2\. \*\*Few-shot effectiveness analysis\*\* (0/3/5-shot comparison)

3\. \*\*Script-handling discovery\*\* (critical for Dravidian languages)

4\. \*\*Comprehensive error taxonomy\*\*

5\. \*\*Reproducible framework\*\* (₹0 budget)



\## 📊 Citation



If you use this code or findings, please cite:

```bibtex

@misc{malayalam-llm-2026,

&#x20; title={Evaluating Large Language Models on Malayalam-English Code-Mixed Sentiment Analysis},

&#x20; author={Your Name},

&#x20; year={2026},

&#x20; note={Research project}

}

```



\## 📝 License



MIT License - See LICENSE file for details



\## 🤝 Contributing



Contributions welcome! Please open an issue or submit a pull request.



\## 📧 Contact



\- \*\*Author\*\*: Your Name

\- \*\*Email\*\*: your.email@example.com

\- \*\*Institution\*\*: Your University



\## 🙏 Acknowledgments



\- DravidianCodeMix dataset creators

\- Groq and Mistral AI for free API access

\- Open-source LLM community



\---



\*\*Status\*\*: 🚧 Research in progress | 📄 Paper submission planned for EMNLP 2026 Workshop

