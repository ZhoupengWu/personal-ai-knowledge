import numpy as np

def cosineSimilarity(a: np.ndarray, b: np.ndarray) -> np.float32:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query_embedding: np.ndarray, chunks_embedding: list[tuple], top_k: int, min_sim: float) -> list[tuple]:
    data_dim = []

    for text, chunk_emd, source, *others in chunks_embedding:
        sim = cosineSimilarity(query_embedding, chunk_emd)
        data_dim.append((text, sim, source, others))

    data_dim_sorted = sorted(data_dim, key=lambda x: x[1], reverse=True)
    filtered = [data for data in data_dim_sorted if data[1] >= min_sim]

    return filtered[:top_k]
