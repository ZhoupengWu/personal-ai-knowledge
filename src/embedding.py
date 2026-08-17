from sentence_transformers import SentenceTransformer

def loadModel(name="paraphrase-multilingual-mpnet-base-v2") -> SentenceTransformer:
    return SentenceTransformer(name)

def embeddedTexts(model: SentenceTransformer, texts: list):
    return model.encode(texts)
