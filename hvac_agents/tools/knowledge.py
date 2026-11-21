"""
knowledge.py - The search_manuals tool used by the Knowledge (RAG) agent.

Built on Cesar's RAG notebook (Area 1): same chunks.json, same embeddings.npy,
same model (all-MiniLM-L6-v2), same MIN_SCORE (0.42) and the same contract:
    question -> up to 4 passages with source and page, or "NO_EVIDENCE".

Search back-ends, in order:
  1. PostgreSQL + pgvector table document_chunks (if DATABASE_URL is set and loaded)
  2. numpy cosine search over embeddings.npy (needs sentence-transformers)
  3. TF-IDF keyword search over chunks.json (only if sentence-transformers is
     missing; scores are on a different scale, so it has its own threshold)
If chunks.json is not in data/, the tool answers NO_EVIDENCE with the reason.
"""
import json
import time

import numpy as np

from .. import config


class ManualLibrary:
    def __init__(self, chunks_file=None, embeddings_file=None, memory=None):
        self.chunks, self.vectors, self.model, self.tfidf = [], None, None, None
        self.backend, self.min_score = "none", config.MIN_SCORE
        self.memory = memory
        cf = chunks_file or config.CHUNKS_FILE
        ef = embeddings_file or config.EMBEDDINGS_FILE
        try:
            with open(cf) as f:
                self.chunks = json.load(f)
        except FileNotFoundError:
            self.load_error = "data/chunks.json not found (copy it from the RAG notebook into the data folder)"
            return
        self.load_error = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(config.EMBED_MODEL)
            self.vectors = np.load(ef)
            if len(self.vectors) != len(self.chunks):
                raise ValueError("embeddings and chunks have different lengths")
            self.backend = "embeddings"
            if memory is not None and memory.kind == "postgres":
                try:
                    memory.query("SELECT COUNT(*) AS n FROM document_chunks")
                    self.backend = "pgvector"
                except Exception:  # table not loaded -> stay with numpy
                    pass
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self.tfidf_matrix = self.tfidf.fit_transform([c["text"] for c in self.chunks])
            self.backend, self.min_score = "tfidf", config.TFIDF_MIN_SCORE

    @property
    def ready(self):
        return bool(self.chunks) and self.backend != "none"

    def _scores(self, question):
        if self.backend in ("embeddings", "pgvector"):
            q = self.model.encode([question], normalize_embeddings=True)[0]
            if self.backend == "pgvector":
                rows = self.memory.query(
                    "SELECT source, page, chunk_text, 1 - (embedding <=> ?::vector) AS score "
                    "FROM document_chunks ORDER BY embedding <=> ?::vector LIMIT ?",
                    (str(list(map(float, q))), str(list(map(float, q))), config.TOP_K))
                return rows
            scores = self.vectors @ q
        else:
            scores = (self.tfidf_matrix @ self.tfidf.transform([question]).T).toarray().ravel()
        best = np.argsort(-scores)[:config.TOP_K]
        return [{"source": self.chunks[i]["source"], "page": self.chunks[i]["page"],
                 "chunk_text": self.chunks[i]["text"], "score": float(scores[i])} for i in best]

    def search_manuals(self, question, retries=2):
        """The tool. Retry twice on errors, then NO_EVIDENCE (never invented sources)."""
        if not self.ready:
            return {"status": "NO_EVIDENCE", "reason": self.load_error or "library not loaded", "passages": []}
        err = None
        for attempt in range(retries + 1):
            try:
                rows = self._scores(question)
                passages = [{"evidence_id": f"{r['source']}#p{r['page']}", "source": r["source"],
                             "page": int(r["page"]), "score": round(float(r["score"]), 3),
                             "text": r["chunk_text"]}
                            for r in rows if float(r["score"]) >= self.min_score]
                if not passages:
                    best = max((float(r["score"]) for r in rows), default=0)
                    return {"status": "NO_EVIDENCE", "reason": f"best score {best:.3f} < {self.min_score}",
                            "passages": [], "backend": self.backend}
                return {"status": "OK", "passages": passages, "backend": self.backend}
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                time.sleep(0.2 * (attempt + 1))
        return {"status": "NO_EVIDENCE", "reason": f"search failed after retries: {err}", "passages": []}
