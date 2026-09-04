from sentence_transformers import SentenceTransformer

MODEL_PREFIXES = {
    "paraphrase-multilingual-mpnet-base-v2": {
        "query": "",
        "passage": ""
    },
    "intfloat/multilingual-e5-large": {
        "query": "query: ",
        "passage": "passage: "
    }
}

def loadModel(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)

def embeddedTexts(model: SentenceTransformer, texts: list, model_name: str, text_type: str):
    prefix = MODEL_PREFIXES.get(model_name, {}).get(text_type, "")
    prefixed = [prefix + t for t in texts]

    return model.encode(prefixed)
