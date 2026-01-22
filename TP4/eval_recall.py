"""
eval_recall.py
Évaluation retrieval-only simple (proxy Recall@k):
On vérifie si le doc_type attendu apparaît dans les top-k résultats.

Usage:
  python TP4/eval_recall.py
"""

import os
import json

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_DIR = os.path.join("TP4", "chroma_db")
COLLECTION_NAME = "tp4_rag"
QUESTIONS_PATH = os.path.join("TP4", "eval", "questions.json")

# Identique à build_index.py
EMBEDDING_MODEL = "nomic-embed-text-v2-moe"

# Comme tes scripts précédents (tu peux tester 3 vs 5)
TOP_K = 5

PORT = "11434"  # Ollama local


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    emb = OllamaEmbeddings(base_url=f"http://127.0.0.1:{PORT}", model=EMBEDDING_MODEL)
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=emb,
        persist_directory=CHROMA_DIR,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    ok = 0
    total = len(dataset)

    print("=" * 80)
    print(f"[EVAL] proxy Recall@{TOP_K} (doc_type attendu dans top-k)")
    print("=" * 80)

    for ex in dataset:
        qid = ex["id"]
        q = ex["question"]
        expected = ex["expected_doc_type"]

        docs = retriever.invoke(q)
        types = [d.metadata.get("doc_type", "unknown") for d in docs]

        hit = expected in types
        ok += 1 if hit else 0

        print(f"\n[{qid}] {q}")
        print(f"  expected: {expected}")
        print(f"  got: {types}")
        print(f"  hit: {hit}")

    score = ok / total if total > 0 else 0.0
    print("\n" + "-" * 80)
    print(f"[SCORE] {ok}/{total} = {score:.2f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
