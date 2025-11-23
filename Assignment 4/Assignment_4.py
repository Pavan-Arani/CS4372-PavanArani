# Colab/local script: NLP project — Text Summarization with Transformers
# Task: Summarize your provided Gutenberg text file

import nltk
import re
import json
from pathlib import Path
from typing import List

from transformers import pipeline, AutoTokenizer
from rouge_score import rouge_scorer

nltk.download("punkt_tab")
nltk.download('punkt')

###################################
# 1) Load your Frankenstein file
###################################
book_path = Path("The_Project_Gutenberg_eBook_of_Frankenstein.txt")

if not book_path.exists():
    raise FileNotFoundError(
        "❌ File not found. Please place 'The_Project_Gutenberg_eBook_of_Frankenstein.txt' in this folder."
    )

raw_text = book_path.read_text(encoding="utf-8")
print("Loaded text length:", len(raw_text))


###################################
# 2) Extract main content
###################################
def extract_main_text(text: str) -> str:
    start = re.search(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*", text, re.IGNORECASE)
    end = re.search(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK .* \*\*\*", text, re.IGNORECASE)

    if start and end:
        content = text[start.end():end.start()]
    else:
        # fallback
        content = text

    content = re.sub(r"\r\n", "\n", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()

main_text = extract_main_text(raw_text)
print("Main content length:", len(main_text))


###################################
# 3) Split into chapters
###################################
def split_into_chapters(text: str) -> List[str]:
    parts = re.split(r"(?i)(?=chapter\s+\w+)", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 400]
    return parts

chapters = split_into_chapters(main_text)

if len(chapters) == 0:
    chunk_size = 15000
    chapters = [main_text[i:i+chunk_size] for i in range(0, len(main_text), chunk_size)]

print("Number of chapters:", len(chapters))


###################################
# 4) Summarizer setup (BART)
###################################
model_name = "facebook/bart-large-cnn"
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn",
    framework="pt"  # FORCE PyTorch
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_TOKENS = 1024


###################################
# 5) Token-based chunking
###################################
def chunk_by_tokens(text, max_tokens=1024, overlap=100):
    ids = tokenizer.encode(text, truncation=False)
    chunks = []

    i = 0
    while i < len(ids):
        j = min(i + max_tokens, len(ids))
        chunk_text = tokenizer.decode(ids[i:j], skip_special_tokens=True)
        chunks.append(chunk_text)

        if j == len(ids):
            break

        i = j - overlap

    return chunks


###################################
# 6) Summarize each chapter
###################################
summaries = []

for idx, chapter in enumerate(chapters):
    print(f"Summarizing chapter {idx+1}/{len(chapters)} ...")

    chunks = chunk_by_tokens(chapter, max_tokens=MAX_TOKENS)
    partial_sums = []

    for ch in chunks:
        try:
            out = summarizer(ch, max_length=200, min_length=50, do_sample=False)
            partial_sums.append(out[0]["summary_text"])
        except:
            partial_sums.append(ch[:700])

    combined = " ".join(partial_sums)

    try:
        refined = summarizer(combined, max_length=250, min_length=100, do_sample=False)[0]["summary_text"]
    except:
        refined = combined

    summaries.append({"chapter": idx + 1, "summary": refined})


###################################
# 7) Final whole-book summary
###################################
full_summary_text = " ".join([s["summary"] for s in summaries])

try:
    final_summary = summarizer(full_summary_text, max_length=350, min_length=150)[0]["summary_text"]
except:
    final_summary = full_summary_text


###################################
# 8) Extractive baseline (replacement for gensim)
###################################
def simple_extractive_summary(text, max_sentences=10):
    sents = nltk.sent_tokenize(text)
    if len(sents) <= max_sentences:
        return text
    long_sents = sorted(sents, key=len, reverse=True)
    return " ".join(long_sents[:max_sentences])

baseline_summary = simple_extractive_summary(main_text, max_sentences=12)


###################################
# 9) ROUGE evaluation
###################################
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(baseline_summary, final_summary)

print("\nROUGE scores:")
print(scores)


###################################
# 10) Save outputs
###################################
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

json.dump(summaries, open(output_dir/"chapter_summaries.json","w"), indent=2)
open(output_dir/"final_summary.txt","w").write(final_summary)
open(output_dir/"baseline_summary.txt","w").write(baseline_summary)
json.dump(scores, open(output_dir/"rouge_scores.json","w"), indent=2)

print("\n✔ All outputs saved to ./outputs/")
print("\nFINAL SUMMARY (truncated):\n")
print(final_summary[:800])