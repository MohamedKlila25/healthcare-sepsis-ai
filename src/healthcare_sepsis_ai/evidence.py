"""Small auditable lexical evidence index. Not a clinical guideline database."""
import re
from collections import Counter
from math import log

import httpx

SOURCES = [
    {"id": "physionet-2019", "title": "PhysioNet Challenge 2019: dataset and target",
     "url": "https://physionet.org/content/challenge-2019/1.0.0/",
     "text": "The challenge provides hourly ICU observations. Its supplied SepsisLabel is shifted six hours before operational sepsis onset. Predicting this label is a challenge-compatible detection task, not an independently derived onset target. The dataset has no linked medical images."},
    {"id": "protocol-splits", "title": "Healthcare SepsisAI evaluation protocol",
     "url": "docs/evaluation.md",
     "text": "Split patients into training, calibration and held-out test sets. Fit imputers and scaling only on training patients. Features use only current and past hours. Evaluate AUPRC, AUROC, Brier score and false alert episodes per nonseptic patient-day. Synthetic metrics do not establish clinical performance."},
    {"id": "model-limitations", "title": "Healthcare SepsisAI model card",
     "url": "docs/model-card.md",
     "text": "The demo uses a calibrated logistic regression trained on synthetic trajectories. High missingness causes abstention. Occlusion contributions measure sensitivity to replacing a feature, not causal clinical explanations. Risk estimates are research outputs, not diagnoses, treatment recommendations or validated clinical probabilities."},
]


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve(query: str, limit: int = 3):
    """BM25-style lexical matching over three authored, attributed research notes."""
    docs = [tokens(s["title"] + " " + s["text"]) for s in SOURCES]
    avg = sum(map(len, docs)) / len(docs)
    ranked = []
    for source, doc in zip(SOURCES, docs):
        counts = Counter(doc)
        score = 0.0
        for term in set(tokens(query)):
            freq = sum(term in d for d in docs)
            tf = counts[term]
            score += log(1 + (len(docs) - freq + 0.5) / (freq + 0.5)) * tf * 2.5 / (
                tf + 1.5 * (0.25 + 0.75 * len(doc) / avg))
        if score > 0:
            ranked.append({**source, "score": round(score, 4), "kind": "research_reference"})
    return sorted(ranked, key=lambda d: d["score"], reverse=True)[:limit]


def search_pubmed(query: str, limit: int = 5):
    """Retrieve live PubMed metadata, not article full text or clinical advice."""
    with httpx.Client(timeout=12) as client:
        response = client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                              params={"db": "pubmed", "term": query, "retmode": "json", "retmax": limit})
        response.raise_for_status()
        ids = response.json()["esearchresult"]["idlist"]
        if not ids:
            return []
        response = client.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                              params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
        response.raise_for_status()
        data = response.json()["result"]
        return [{"id": pmid, "title": data[pmid]["title"], "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                 "kind": "pubmed_metadata", "publication_date": data[pmid].get("pubdate")} for pmid in ids]

