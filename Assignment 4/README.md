# Assignment 4 – NLP Text Summarization with Transformers

## Overview
This project performs **text summarization** on the novel *Frankenstein* using a **Transformer-based abstractive model (BART-Large CNN)**. The dataset consists of a single `.txt` file from Project Gutenberg. The project preprocesses the text, splits it into chapters, generates chapter-level summaries, produces a full novel summary, and evaluates the results using **ROUGE-1, ROUGE-2, and ROUGE-L** metrics.

---

## Requirements
- Python 3.10+  
- PyTorch (with MPS support on macOS for GPU acceleration)  
- Transformers (HuggingFace)  
- NLTK  

Install dependencies:

```bash
pip install torch transformers nltk