import os
import sys
import time
import argparse
from pathlib import Path
from itertools import chain
from datetime import datetime, timezone
from dotenv import load_dotenv
from storage import createConnection, createTableChunk, insertChunk, getChunksByModel, deleteChunksBySource, createTableLog, logQuery, saveAnswerToFile
from chunking import chunkText, chunkTextBySentence
from embedding import loadModel, embeddedTexts
from search import search
from generation import createClient, generateAnswer
from readers import readFile
from evaluation import loadQuestions, groupByCategory, scoreQuestions, summarize, printReport

CATEGORY_MODELS = {
    "note": {
        "model": "paraphrase-multilingual-mpnet-base-v2",
        "min_sim": 0.4
    },
    "programma": {
        "model": "intfloat/multilingual-e5-large",
        "min_sim": 0.8
    }
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
    default="note",
    help="Categoria del contenuto: determina il modello di embedding usato per indicizzare i file "
         "(note: testi brevi in prosa; programma: documenti lunghi, come un programma politico). "
         "Usa la stessa categoria in fase di query (default: note)."
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
    choices=["strict", "standard", "full"],
    default="strict",
    help="Quanto la risposta può discostarsi dalle informazioni fornite: 'strict' usa solo quelle, "
         "'standard' le integra con conoscenza generale segnalandolo sempre, "
         "'full' integra liberamente senza distinzioni esplicite."
)

query_parser.add_argument(
    "--min-sim",
    type=float,
    default=None,
    help="Soglia minima di similarità (0-1) sotto la quale un chunk viene scartato prima di essere passato al modello. "
         "Se omessa, si usa quella calibrata per la categoria. "
         "Indica 0 per vedere tutti i risultati, utile con --show-chunks per calibrare. "
         "Le soglie non sono confrontabili tra categorie diverse perché ogni modello ha una scala di punteggi propria."
)

query_parser.add_argument(
    "--category",
    choices=list(CATEGORY_MODELS.keys()),
    default="note",
    help="Categoria su cui cercare: deve coincidere con quella usata in fase di indicizzazione "
         "perché modelli diversi producono embedding non confrontabili (default: note)."
)

query_parser.add_argument(
    "--show-chunks",
    action="store_true",
    help="Stampa i chunks recuperati (testo, punteggio di similarità e fonte) prima della risposta. "
         "Utile per verificare cosa riceve il modello e per calibrare --min-sim."
)

query_parser.add_argument(
    "--temperature",
    type=float,
    default=None,
    help="Temperatura di generazione (0 = risposte più stabili e ripetibili, valori più alti = più variabili). "
         "Se omessa, dipende dalla modalità: strict 0.0, standard 0.3, full 0.6."
)

#
# eval command
#
eval_parser = subparsers.add_parser(
    "eval",
    help="Valuta le soglie di similarità su un insieme di domande etichettate senza chiamare l'LLM.",
    description="Per ogni domanda del file JSON cerca i chunks più simili e riassume come si separano le domande pertinenti da quelle fuori tema, "
         "per verificare o calibrare la soglia di ogni categoria. Non usa l'api: serve solo per la ricerca."
)

eval_parser.add_argument(
    "file",
    help="Percorso del file JSON con le domande. Ogni elemento ha: question, category, label (pertinente, borderline o fuori_tema) "
         "e, facoltativo, expected_source (file da cui ci si aspetta il primo risultato)."
)

eval_parser.add_argument(
    "--top-k",
    type=int,
    default=5,
    help="Numero di chunk analizzati per ogni domanda: tutti i loro punteggi entrano nel riepilogo (default: 5)."
)

eval_parser.add_argument(
    "--verbose",
    action="store_true",
    help="Stampa anche punteggio e fonte di ogni chunk recuperato per ogni domanda."
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
    hf_model_name = CATEGORY_MODELS[args.category]["model"]

    model = loadModel(hf_model_name)
    conn_chunk = createConnection("test.db")
    createTableChunk(conn_chunk)

    for file_path in files:
        deleteChunksBySource(conn_chunk, file_path.name)
        text = readFile(file_path)
        chunked_text = chunkText(text, args.dimension, args.overlap) if strategy == "word" else chunkTextBySentence(text, args.dimension, args.overlap)
        embed_chunk = embeddedTexts(model, chunked_text, hf_model_name, "passage")

        for i in range(len(chunked_text)):
            insertChunk(conn_chunk, chunked_text[i], embed_chunk[i], file_path.name, hf_model_name)

        counter += 1
        print(f"[{counter}/{total_counter}] file indexed ({file_path.name})")

    print("DONE")
elif args.command == "query":
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

    if api_key is None:
        print("Manca l'api key")

        sys.exit(1)

    query = args.text
    mode = args.mode
    category = args.category
    hf_model_name = CATEGORY_MODELS[category]["model"]
    min_sim = args.min_sim if args.min_sim is not None else CATEGORY_MODELS[category]["min_sim"]

    model = loadModel(hf_model_name)

    conn_chunk = createConnection("test.db")
    createTableChunk(conn_chunk)

    conn_log = createConnection("logs_v2.db")
    createTableLog(conn_log)

    client = createClient(api_key)

    embed_query = embeddedTexts(model, [query], hf_model_name, "query")
    chunks = getChunksByModel(conn_chunk, hf_model_name)

    if chunks is None:
        print("Non ci sono valori per questa ricerca")

        sys.exit(0)

    start_time = time.time()
    result = search(embed_query[0], chunks, args.top_k, min_sim)

    if args.show_chunks:
        print(result)

    if not result and mode == "strict":
        print("Nessuna informazione pertinente è stata trovata nelle note...")

        sys.exit(0)

    sources_set = set([a[2] for a in result])
    sources: str = ", ".join(source for source in sources_set)
    answer_text, usage = generateAnswer(client, api_model_name, result, query, mode, args.temperature)
    elapsed_time = time.time() - start_time

    timestamp = datetime.now(timezone.utc).isoformat()
    input_tokens = usage.prompt_tokens
    input_cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
    output_tokens = usage.completion_tokens
    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details and usage.completion_tokens_details.reasoning_tokens is not None else 0
    total_tokens = usage.total_tokens

    logQuery(conn_log, hf_model_name, timestamp, query, args.category, len(result), sources, mode, api_model_name, input_tokens, input_cached_tokens, output_tokens, reasoning_tokens, total_tokens, elapsed_time)

    print(f"\n{answer_text}")
    print(f"\nFONTI CONSULTATE: [{sources if sources else "nessuna"}]")
    print("\n=====")
    print(f"INPUT = {input_tokens} token (CACHED = {input_cached_tokens} token)\nOUTPUT = {output_tokens} token (REASONING = {reasoning_tokens} token)\nTOTAL = {total_tokens} token")
    print("=====")

    saveAnswerToFile(query, answer_text, timestamp, sources, mode, category)
elif args.command == "eval":
    try:
        questions = loadQuestions(args.file, list(CATEGORY_MODELS.keys()))
    except (OSError, ValueError) as e:
        print(f"Impossibile leggere le domande: {e}")

        sys.exit(1)

    conn_chunk = createConnection("test.db")
    createTableChunk(conn_chunk)

    for category, group in groupByCategory(questions).items():
        hf_model_name = CATEGORY_MODELS[category]["model"]
        threshold = CATEGORY_MODELS[category]["min_sim"]
        chunks = getChunksByModel(conn_chunk, hf_model_name)

        if chunks is None:
            print(f"Categoria {category}: nessun chunk indicizzato, viene saltato")

            continue

        model = loadModel(hf_model_name)
        results = scoreQuestions(model, hf_model_name, chunks, group, args.top_k)
        summary = summarize(results, threshold)
        printReport(category, results, summary, args.verbose)
else:
    parser.print_help()
