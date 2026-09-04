import os
import sys
import argparse
from pathlib import Path
from itertools import chain
from dotenv import load_dotenv
from storage import createConnection, createTable, insertChunk, getAllChunks, deleteChunksBySource
from chunking import chunkText, chunkTextBySentence
from embedding import loadModel, embeddedTexts
from search import search
from generation import createClient, generateAnswer
from readers import readFile

CATEGORY_MODELS = {
    "note": "paraphrase-multilingual-mpnet-base-v2",
    "programma": "intfloat/multilingual-e5-large"
}

parser = argparse.ArgumentParser(description="Personal AI Knowledge - indicizza le tue note e fai domande basate sul loro contenuto.")
subparsers = parser.add_subparsers(dest="command", help="Comando da eseguire.")

#
# index command
#
index_parser = subparsers.add_parser(
    "index",
    help="Indicizza i file .md e .pdf di una cartella: li divide in chunk, genera gli embedding e li salva nel database.",
    description="Indicizza i file .md e .pdf di una cartella: li divide in chunk, genera gli embedding e li salva nel database.",
)

index_parser.add_argument(
    "folder",
    help="Percorso della cartella contenente i file .md e/o .pdf da indicizzare.",
)

index_parser.add_argument(
    "--dimension",
    type=int,
    default=40,
    help="Dimensione del chunk. Con --strategy=word: numero esatto di parole per chunk. "
         "Con --strategy=sentence: numero massimo di parole per chunk (il chunk può superare leggermente questo limite "
         "per includere l'ultima frase intera)."
)

index_parser.add_argument(
    "--overlap",
    type=int,
    default=1,
    help="Sovrapposizione tra chunk consecutivi, per non prendere contesto ai margini. "
         "Con --strategy=word: espresso in numero di parole. "
         "Con --strategy=sentence: espresso in numero di frasi."
)

index_parser.add_argument(
    "--strategy",
    choices=["word", "sentence"],
    default="sentence",
    help="Metodo di chunking: 'word' taglia per numero fisso di parole (più semplice, può spezzare le frasi); "
         "'sentence' raggruppa frasi intere fino al limite di parole, senza spezzarle (default, consigliato per note in prosa)."
)

index_parser.add_argument(
    "--category",
    choices=list(CATEGORY_MODELS.keys()),
    default="programma"
)

#
# query command
#
query_parser = subparsers.add_parser(
    "query",
    help="Cerca nelle note indicizzate e genera una risposta basata sui chunk più pertinenti.",
    description="Cerca nelle note indicizzate e genera una risposta basata sui chunk più pertinenti."
)

query_parser.add_argument(
    "text",
    help="La domanda da porre, basata sulle note indicizzate."
)

query_parser.add_argument(
    "--top-k",
    type=int,
    default=5,
    help="Numero massimo di chunk da recuperare e passare al modello come contesto (default: 5)."
)

query_parser.add_argument(
    "--mode",
    choices=["strict", "hybrid"],
    default="strict",
    help="'strict': risponde solo con informazioni presenti nelle note, dice esplicitamente se non le trova. "
         "'hybrid': non ancora disponibile, ricade automaticamente su 'strict'."
)

query_parser.add_argument(
    "--min-sim",
    type=float,
    default=0.4,
    help="Soglia minima di similarità (0-1) sotto la quale un chunk viene scartato prima di essere passato al modello. "
         "Serve solo a filtrare il rumore più evidente, non è una soglia di rilevanza precisa (default: 0.4)."
)


args = parser.parse_args()

if args.command == "index":
    folder = Path(args.folder)

    if not folder.exists():
        print("La cartella non esiste")

        sys.exit(1)

    if not folder.is_dir():
        print("Non è una cartella")

        sys.exit(1)

    files = list(chain(folder.glob("*.md"), folder.glob("*.pdf")))
    total_counter = len(files)

    if total_counter == 0:
        print("Non ci sono file da indicizzare")

        sys.exit(1)

    counter = 0
    strategy = args.strategy
    model_name = CATEGORY_MODELS[args.category]

    model = loadModel(model_name)
    conn = createConnection("test.db")
    createTable(conn)

    for file_path in files:
        deleteChunksBySource(conn, file_path.name)
        text = readFile(file_path)
        chunked_text = chunkText(text, args.dimension, args.overlap) if strategy == "word" else chunkTextBySentence(text, args.dimension, args.overlap)
        embed_chunk = embeddedTexts(model, chunked_text, model_name, "passage")

        for i in range(len(chunked_text)):
            insertChunk(conn, chunked_text[i], embed_chunk[i], file_path.name, model_name)

        counter += 1
        print(f"[{counter}/{total_counter}] file indexed ({file_path.name})")

    print("DONE")
elif args.command == "query":
    load_dotenv()

    api_key = os.getenv("API_KEY")

    if api_key is None:
        print("Manca l'api key")

        sys.exit(1)

    query = args.text

    if args.mode == "hybrid":
        print("Modalità hybrid non disponibile. Verrà utilizzato strict")

    mode = "strict" if args.mode == "hybrid" else args.mode

    model = loadModel()
    conn = createConnection("test.db")
    createTable(conn)

    client = createClient(api_key)

    embed_query = embeddedTexts(model, [query])
    chunks = getAllChunks(conn)

    if chunks is None:
        print("Il db è vuoto")

        sys.exit(1)

    result = search(embed_query[0], chunks, args.top_k, args.min_sim)

    if not result:
        print("Nessuna informazione pertinente è stata trovata nelle note...")

        sys.exit(0)

    sources = set([a[2] for a in result])
    answer = generateAnswer(client, result, query, mode=mode)

    print(answer[0])
    print(f"FONTI: [{", ".join(source for source in sources)}]")
    print("\n=====")
    print(f"INPUT = {answer[1].prompt_tokens} token (CACHED = {answer[1].prompt_tokens_details.cached_tokens} token)\nOUTPUT = {answer[1].completion_tokens} token\nTOTAL = {answer[1].total_tokens} token")
    print("=====")
else:
    parser.print_help()
