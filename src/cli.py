import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from storage import createConnection, createTable, insertChunk, getAllChunks, deleteChunksBySource
from chunking import chunkText
from embedding import loadModel, embeddedTexts
from search import search
from generation import createClient, generateAnswer

parser = argparse.ArgumentParser(description="Indicizza o ricerca qualcosa")
subparsers = parser.add_subparsers(dest="command")

index_parser = subparsers.add_parser("index")
index_parser.add_argument("folder")
index_parser.add_argument("--dimension", type=int, default=50)
index_parser.add_argument("--overlap", type=int, default=10)

query_parser = subparsers.add_parser("query")
query_parser.add_argument("text")
query_parser.add_argument("--top-k", type=int, default=5)
query_parser.add_argument("--mode", choices=["strict", "hybrid"], default="strict")

args = parser.parse_args()

counter = 0

if args.command == "index":
    folder = Path(args.folder)

    if not folder.exists():
        print("La cartella non esiste")

        sys.exit(1)

    if not folder.is_dir():
        print("Non è una cartella")

        sys.exit(1)

    total_counter = len(list(folder.glob("*.md")))

    if total_counter == 0:
        print("Non ci sono file da indicizzare")

        sys.exit(1)

    model = loadModel()
    conn = createConnection("test.db")
    createTable(conn)

    for file_path in folder.glob("*.md"):
        deleteChunksBySource(conn, file_path.name)
        text = file_path.read_text(encoding="utf-8")
        chunked_text = chunkText(text, args.dimension, args.overlap)
        embed_chunk = embeddedTexts(model, chunked_text)

        for i in range(len(chunked_text)):
            insertChunk(conn, chunked_text[i], embed_chunk[i], file_path.name)

        counter += 1
        print(f"[{counter}/{total_counter}] file indexed")

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

    result = search(embed_query[0], chunks, args.top_k)
    sources = set([a[2] for a in result])
    answer = generateAnswer(client, result, query, mode=mode)

    print(answer[0])
    print(f"FONTI: [{" ".join(source for source in sources)}]")
    print("\n=====")
    print(f"INPUT = {answer[1].prompt_tokens} token (CACHED = {answer[1].prompt_tokens_details.cached_tokens} token)\nOUTPUT = {answer[1].completion_tokens} token\nTOTAL = {answer[1].total_tokens} token")
    print("=====")
else:
    parser.print_help()
