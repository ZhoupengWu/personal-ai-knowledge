def chunkText(text: str, dimension: int, overlap: int) -> list:
    if overlap >= dimension:
        raise ValueError("L'overlap non può essere uguale o più grande della dimensione")

    list_word = text.split()
    chunk = []

    for i in range(0, len(list_word), dimension - overlap):
        chunk.append(" ".join(list_word[i : i + dimension]))

    return chunk
