### 📄 `README.md`
# 🔣 tokeniser-py

**Imp Links: [PyPI Library](https://pypi.org/project/tokeniser-py/) | [PyPI Lite Library (tokeniser-py-lite)](https://pypi.org/project/tokeniser-py-lite/) | [Lite Library GitHub (tokeniser-py-lite)](https://github.com/Tasmay-Tibrewal/tokeniser-py-lite) | [Demo (HF Spaces)](https://huggingface.co/spaces/Tasmay-Tib/Tokeniser-py) | [Complete repo (unchunked) - HF](https://huggingface.co/datasets/Tasmay-Tib/Tokeniser) | [Complete repo (chunked) - GitHub](https://github.com/Tasmay-Tibrewal/Tokeniser) | [Imp Files Github](https://github.com/Tasmay-Tibrewal/Tokeniser-imp)**

A high-performance, fully custom tokeniser built from scratch — no BPE, no existing NLP tokenisation scheme. This tokeniser is based on a unique algorithm developed independently and trained on over **1 billion tokens** from the **SlimPajama dataset** (Val + Test), providing an efficient, interpretable, and extendable tokenisation pipeline.

## 🚀 What This Library Offers

- Tokeniser built on a vocabulary of **131,072 tokens**
- Two versions of vocab:
  - `0.5B`: Validation-only data
  - `1B`: Validation + Test data
- Token vocab built via a **custom algorithm** — no Byte Pair Encoding (BPE)
- Tokenisation logic includes:
  - Token lookup from pre-generated token map
  - Dynamic programming-based segmentation for out-of-vocab tokens
  - One-hot encoding (NumPy or PyTorch)
  - Visualisation utilities for tokens and token IDs
- Lightweight JSON format for token maps & token count maps
- Ready for integration into any LLM pre-tokenisation pipeline

> Note: Files (chunked less than 2GB) are stored on [Hugging Face](https://huggingface.co/) instead of [GitHub](https://github.com/) due to LFS file size constraints. On [GitHub](https://github.com/) (files chunked below 100MB) are available.

## 📦 Installation
```bash
pip install tokeniser-py
```

## 🛠 Usage
```python
from tokeniser import Tokeniser

t = Tokeniser()
tokens, count = t.tokenise("Your input text here.")
token_ids = t.token_ids(tokens)
```

Use `t.one_hot_tokens(token_ids)` for NumPy-based one-hot encoding, or `op='torch'` for PyTorch.

## 📚 Data Sources

All token maps and token counts are generated from the [SlimPajama dataset](https://huggingface.co/datasets/cerebras/SlimPajama-627B) by Cerebras.

## 📁 Vocab Files
- `ordered_tokenizer_1b_val_test_data.json` — Ordered tokens (1B data)
- `unordered_tokenizer_1b_val_test_data.json` — Unordered tokens (1B)
- `count_tokenizer_1b_val_test_data.json` — Token counts (1B)
- (Similar structure for 0.5B val-only version)

## 📌 Design Philosophy

This tokeniser is **built from scratch** before learning existing algorithms like BPE. It is designed with the intent to **understand, innovate**, and compare with existing solutions from first principles.

> Some parts may overlap with BPE/WordPiece in spirit — but the core algorithm was independently designed.

## 🤝 Contributions

Feel free to contribute anything via GitHub.

## 📖 License

MIT License