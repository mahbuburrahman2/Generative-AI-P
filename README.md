# Generative-AI-P

A practical **learning and development** repository for exploring modern **Generative AI** concepts and implementations with **Python**.

---

## 1) Project Overview

**Generative-AI-P** is designed as a hands-on experimentation space for developers and AI/ML researchers who want to understand how modern Generative AI systems are built and evaluated.  
This project focuses on practical examples, iterative experimentation, and reproducible learning workflows.

> ⚠️ This is **not** a production-ready system. It is a practical repository for learning, experimentation, and development.

---

## 2) Objectives

- Build practical intuition around LLM-based workflows
- Experiment with prompt engineering patterns
- Compare cloud and local embedding approaches
- Implement semantic similarity pipelines
- Develop AI application prototypes in Python
- Strengthen reproducible development habits using virtual environments

---

## 3) Technologies and Tools

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| LLM / APIs | Google Gemini API |
| Frameworks | LangChain |
| Model Ecosystem | Hugging Face |
| Embeddings | Gemini Embeddings, Sentence Transformers, Local Embeddings |
| Core Tasks | Semantic Similarity, Batch Embeddings |
| Environment | Python virtual environments (`venv`) |

---

## 4) Topics Covered

- Large Language Models (LLMs)
- Prompt Engineering
- LangChain basics and integrations
- Hugging Face model usage
- Google Gemini API usage
- Gemini Embeddings
- Local text embeddings
- Sentence Transformers
- Semantic similarity workflows
- Batch embedding pipelines
- AI application development patterns

---

## 5) Project Structure

Current repository structure:

```text
Generative-AI-P/
└── README.md
```

As the project grows, practical scripts/notebooks and supporting files can be organized into folders such as:

- `examples/` for runnable scripts
- `notebooks/` for exploratory workflows
- `data/` for sample datasets (non-sensitive)
- `src/` for reusable utility modules

---

## 6) Installation and Setup

### Prerequisites

- Python 3.10+
- `pip`

### Setup Steps

```bash
# 1. Clone repository
git clone https://github.com/mahbuburrahman2/Generative-AI-P.git
cd Generative-AI-P

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

# 4. Install dependencies (when requirements are available)
# pip install -r requirements.txt
```

---

## 7) Environment Variables / API Key Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Then load it in Python (example):

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
```

✅ Keep secrets out of source control and never hardcode API keys in scripts.

---

## 8) How to Run the Examples

Run Python example scripts from the project root:

```bash
python path/to/your_example_script.py
```

If using notebooks:

```bash
jupyter notebook
```

Then open and run the desired notebook interactively.

---

## 9) Example Use Cases

- Compare embeddings from Gemini and local sentence-transformer models
- Build a semantic similarity search experiment for text matching
- Prototype prompt templates for LLM tasks
- Create small AI utilities using LangChain + external model APIs
- Test batch embedding strategies for larger text collections

---

## 10) Learning Outcomes

By working through this repository, you can learn to:

- Design and evaluate prompt-based workflows
- Integrate and test multiple LLM/embedding providers
- Build semantic similarity pipelines end to end
- Structure repeatable AI experiments in Python
- Transition from prototype exploration toward stronger engineering practices

---

## 12) Future Improvements

- Add structured examples by topic (LLMs, embeddings, LangChain)
- Add reproducible notebooks and benchmark comparisons
- Add lightweight test coverage for reusable utility modules
- Add dependency and environment management files
- Add CI checks for code quality and basic validation

---

## 13) Author

**Mahbubur Rahman**  
GitHub: [@mahbuburrahman2](https://github.com/mahbuburrahman2)

---

If you are learning Generative AI with Python, feel free to fork this repo and build your own experiments 🚀
