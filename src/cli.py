import argparse
from pathlib import Path
from storage import createConnection, createTable, insertChunk, getAllChunks
from chunking import chunkText
from embedding import loadModel, embeddedTexts
from search import search

parser = argparse.ArgumentParser(description="Indicizza o ricerca qualcosa")
subparsers = parser.add_subparsers(dest="command")

index_parser = subparsers.add_parser("index")
index_parser.add_argument("folder")

query_parser = subparsers.add_parser("query")
query_parser.add_argument("text")
query_parser.add_argument("--top-k", type=int, default=5)

args = parser.parse_args()

counter = 0

if args.command == "index":
    model = loadModel()
    conn = createConnection("test.db")
    createTable(conn)
    
    folder = Path(args.folder)
    total_counter = len(list(folder.glob("*.md")))

    for file_path in folder.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")
        chunked_text = chunkText(text, 50, 10)
        embed_chunk = embeddedTexts(model, chunked_text)

        for i in range(len(chunked_text)):
            insertChunk(conn, chunked_text[i], embed_chunk[i], file_path.name)

        counter += 1
        print(f"[{counter}/{total_counter}] file indexed")

    print("DONE")
elif args.command == "query":
    model = loadModel()
    conn = createConnection("test.db")
    createTable(conn)

    embed_query = embeddedTexts(model, [args.text])
    chunks = getAllChunks(conn)
    result = search(embed_query[0], chunks, args.top_k)

    for text, sim, source in result:
        print(f"{sim} ---> {text[:100]}... ({source})")
else:
    parser.print_help()
