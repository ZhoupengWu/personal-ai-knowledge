import numpy as np

def cosineSimilarity(a, b) -> np.float32:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query_embedding, chunks_embedding: list[tuple], top_k = 5) -> list[tuple]:
    query_embedding = list(query_embedding)

    data_dim = []

    for i in range(len(chunks_embedding)):
        sim = cosineSimilarity(query_embedding[0], chunks_embedding[i][1])
        data_dim.append((chunks_embedding[i][0], sim, chunks_embedding[i][2]))
    
    data_dim_sorted = sorted(data_dim, key=lambda x: x[1], reverse=True)

    return data_dim_sorted[0 : top_k]
