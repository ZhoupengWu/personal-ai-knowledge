import numpy as np
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

MAX_LENGTH = 1024

def loadReranker(name: str = RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(name, max_length=MAX_LENGTH)

def toProbability(scores) -> np.ndarray:
    # A seconda della versione di sentence-transformers il modello restituisce già un valore tra 0 e 1
    # (sigmoide applicata di default) oppure il logit grezzo. In quest'ultimo caso si applica la sigmoide,
    # così i punteggi sono sempre nella stessa scala, tra 0 e 1.
    scores = np.asarray(scores, dtype=np.float64)

    if scores.min() < 0 or scores.max() > 1:
        scores = 1 / (1 + np.exp(-scores))

    return scores

def rerank(reranker: CrossEncoder, question: str, chunks: list[tuple], top_k: int) -> list[tuple]:
    # chunks ha il formato restituito da search: (testo, similarità, fonte, altri campi).
    # Il risultato ha lo stesso formato, ma la similarità è sostituita dal punteggio del re-ranker
    # e i chunk sono riordinati dal più pertinente.
    if not chunks:
        return []

    pairs = [(question, chunk[0]) for chunk in chunks]
    scores = toProbability(reranker.predict(pairs))
    ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)

    return [(chunk[0], float(score), *chunk[2:]) for chunk, score in ranked[:top_k]]
