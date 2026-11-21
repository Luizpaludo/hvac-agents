"""
build_manual_library.py - Rebuilds data/chunks.json (and, optionally, data/embeddings.npy)
from the 3 PDF manuals, with exactly the method of Cesar's RAG notebook (Parts 2, 3, 7 and 8).

Area 1: RAG manual library (owner: Cesar Antonio Noriega Martinez)

Usage:
  python scripts/build_manual_library.py                 # chunks.json only
  python scripts/build_manual_library.py --embeddings    # also embeddings.npy (needs sentence-transformers)

Put the PDFs in data/manuals/ (any names that contain these IDs):
  NIST IR 7365      https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7365.pdf
  Trane CAH-SVX01A  https://archive.org/download/manualsonline-id-aeb01d19-71a1-46b2-a01c-071ff272df15/aeb01d19-71a1-46b2-a01c-071ff272df15.pdf
  ASHRAE 1020-RP    https://core.ac.uk/download/288388245.pdf
The PDFs are not stored in Git (copyright); only the chunk file is used by the agent.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
MANUALS = ROOT / "data" / "manuals"
OUT = ROOT / "data"

# label -> (words that identify the file, first page, last page)  (1 = first page of the PDF)
KEYWORDS = {
    "NIST": (["nist", "7365"], 1, 54),               # skip the code appendix
    "TRANE": (["trane", "aeb01d19", "cah-svx"], 47, None),   # maintenance and troubleshooting only
    "ASHRAE": (["ashrae", "288388245", "1020", "2813"], 1, None),
}
CHUNK_SIZE, OVERLAP = 800, 150


def clean_name(name):
    n = name.lower()
    if "nist" in n or "7365" in n:
        return "nist_nistir7365_ahu_fdd.pdf"
    if "trane" in n or "aeb01d19" in n or "cah-svx" in n:
        return "trane_cah-svx01a-en_air_handlers.pdf"
    if "ashrae" in n or "ashare" in n or "288388245" in n or "1020" in n or "2813" in n:
        return "ashrae_1020rp_ahu_fdd_paper.pdf"
    return name


def main(make_embeddings=False):
    files = sorted(p.name for p in MANUALS.glob("*.pdf"))
    pages = []
    for label, (words, first, last) in KEYWORDS.items():
        match = [n for n in files if any(w in n.lower() for w in words)]
        if not match:
            sys.exit(f"MISSING FILE for {label} in {MANUALS}")
        reader = PdfReader(MANUALS / match[0])
        total = len(reader.pages)
        last = last or total
        kept = 0
        for number in range(first, min(last, total) + 1):
            text = " ".join((reader.pages[number - 1].extract_text() or "").split())
            if len(text) > 50:
                pages.append({"source": clean_name(match[0]), "page": number, "text": text})
                kept += 1
        print(f"{label}: {match[0]} -> {total} pages, {kept} kept")
    chunks = []
    for p in pages:
        start = 0
        while start < len(p["text"]):
            piece = p["text"][start:start + CHUNK_SIZE]
            if len(piece) > 100:
                chunks.append({"source": p["source"], "page": p["page"], "text": piece})
            start += CHUNK_SIZE - OVERLAP
    print("Total chunks:", len(chunks), dict(Counter(c["source"] for c in chunks)))
    (OUT / "chunks.json").write_text(json.dumps(chunks))
    if make_embeddings:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        vectors = SentenceTransformer("all-MiniLM-L6-v2").encode(
            [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
        np.save(OUT / "embeddings.npy", vectors)
        print("embeddings:", vectors.shape)


if __name__ == "__main__":
    main("--embeddings" in sys.argv)
