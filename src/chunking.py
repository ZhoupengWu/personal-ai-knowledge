import re

def chunkText(text: str, dimension: int, overlap: int) -> list:
    if overlap >= dimension:
        raise ValueError("L'overlap non può essere uguale o più grande della dimensione")

    list_word = text.split()
    chunk = []

    for i in range(0, len(list_word), dimension - overlap):
        chunk.append(" ".join(list_word[i : i + dimension]))

    return chunk

def chunkTextBySentence(text: str, max_words: int, overlap_senteces: int) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunk_sentences = []
    covered_until = 0
    counter = 0
    k = 0

    for i in range(0, len(sentences)):
        counter += len(sentences[i].split())

        if counter >= max_words:
            chunk_sentences.append(" ".join(sentences[k : i+1]))
            k = i - overlap_senteces + 1 if i - overlap_senteces + 1 >= 0 else 0
            covered_until = i + 1
            counter = sum(len(a.split()) for a in sentences[k : i+1])

    if covered_until < len(sentences):
        chunk_sentences.append(" ".join(sentences[k : len(sentences)]))

    return chunk_sentences
