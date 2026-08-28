import numpy as np

def cosineSimilarity(a: np.ndarray, b: np.ndarray) -> np.float32:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query_embedding: np.ndarray, chunks_embedding: list[tuple], top_k = 5, min_sim = 0.4) -> list[tuple]:
    data_dim = []

    for text, chunk_emd, source in chunks_embedding:
        sim = cosineSimilarity(query_embedding, chunk_emd)
        data_dim.append((text, sim, source))

    data_dim_sorted = sorted(data_dim, key=lambda x: x[1], reverse=True)
    filtered = [data for data in data_dim_sorted if data[1] >= min_sim]

    return filtered[:top_k]
