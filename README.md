# Personal AI Knowledge

Sistema RAG (Retrieval-Augmented Generation) per indicizzare note personali e documenti e interrogarli in linguaggio naturale. L'indicizzazione e la ricerca semantica girano in locale; la risposta finale viene generata da un modello linguistico via API (attualmente DeepSeek), a cui vengono passati solo i frammenti di testo recuperati per quella domanda.

## Indice

- [Personal AI Knowledge](#personal-ai-knowledge)
  - [Indice](#indice)
  - [Caratteristiche](#caratteristiche)
  - [Come funziona](#come-funziona)
  - [Requisiti](#requisiti)
  - [Installazione](#installazione)
  - [Configurazione](#configurazione)
  - [Utilizzo](#utilizzo)
    - [Indicizzare documenti](#indicizzare-documenti)
    - [Fare domande](#fare-domande)
    - [Modalità di risposta](#modalità-di-risposta)
    - [Output](#output)
  - [Struttura del progetto](#struttura-del-progetto)
  - [Dettagli tecnici](#dettagli-tecnici)
    - [Chunking](#chunking)
    - [Embedding per categoria](#embedding-per-categoria)
    - [Retrieval](#retrieval)
    - [Generazione](#generazione)
    - [Log](#log)
  - [Privacy](#privacy)
  - [Limiti noti](#limiti-noti)
  - [Roadmap](#roadmap)

## Caratteristiche

- Indicizzazione di file **Markdown (.md)** e **PDF (.pdf)**
- Chunking per frasi (non spezza le frasi a metà) con overlap configurabile, oppure per numero fisso di parole
- Embedding multilingua con **modello diverso per categoria di contenuto** (`note`, `programma`)
- Storage locale su **SQLite**
- Ricerca semantica per similarità coseno con soglia minima configurabile
- **Tre modalità di risposta** (`strict`, `standard`, `full`) che regolano quanto il modello può uscire dalle fonti indicizzate
- Elenco delle fonti consultate per ogni risposta
- Log strutturato di ogni query (modalità, fonti, token, tempi) e risposta salvata su file di testo

## Come funziona

**Indicizzazione** (`index`):

```
File (.md/.pdf) → lettura → chunking → embedding → SQLite
```

**Interrogazione** (`query`):

```
Domanda → embedding → ricerca per similarità (top-k, soglia) → LLM → risposta + fonti + log
```

## Requisiti

- Python 3.12+
- Un account DeepSeek con API key, o un altro provider con API compatibile OpenAI (da adattare in `generation.py`)
- Una **GPU è fortemente consigliata per l'indicizzazione**: con il modello della categoria `programma` (`multilingual-e5-large`) l'indicizzazione di circa 300 chunk su CPU non finiva in 5 minuti, mentre su una GPU T4 (Google Colab) richiede pochi secondi. Per le query basta la CPU.

Dipendenze Python principali: `sentence-transformers`, `numpy`, `pypdf`, `openai`, `python-dotenv`.

## Installazione

```bash
git clone https://github.com/ZhoupengWu/personal-ai-knowledge.git
cd personal-ai-knowledge

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configurazione

Copia `.env.example` in `.env` e compila i valori:

```
DEEPSEEK_API_KEY=la-tua-api-key
DEEPSEEK_MODEL=deepseek-flash
```

`DEEPSEEK_MODEL` è opzionale (il default nel codice è lo stesso valore di `.env.example`).

## Utilizzo

### Indicizzare documenti

```bash
python src/cli.py index <cartella> [opzioni]
```

| Opzione | Default | Descrizione |
|---|---|---|
| `folder` | (obbligatorio) | Cartella con i file `.md` e/o `.pdf` da indicizzare (non ricorsivo) |
| `--category` | `note` | Categoria del contenuto, determina il modello di embedding |
| `--strategy` | `sentence` | `sentence`: raggruppa frasi intere; `word`: taglia ogni N parole |
| `--dimension` | `40` | Tetto di parole per chunk (vedi [Chunking](#chunking)) |
| `--overlap` | `1` | Sovrapposizione tra chunk: frasi con `sentence`, parole con `word` |

Esempio:

```bash
python src/cli.py index ./documenti/programma --category programma
```

Rilanciare `index` su una cartella già indicizzata è sicuro: i chunk di ogni file vengono cancellati e ricreati, senza duplicati.

### Fare domande

```bash
python src/cli.py query "<domanda>" [opzioni]
```

| Opzione | Default | Descrizione |
|---|---|---|
| `text` | (obbligatorio) | La domanda |
| `--category` | `note` | Categoria su cui cercare, la stessa usata in indicizzazione |
| `--top-k` | `5` | Numero massimo di chunk passati al modello |
| `--min-sim` | `0.4` | Soglia minima di similarità (vedi [Limiti noti](#limiti-noti)) |
| `--mode` | `strict` | Modalità di risposta (vedi sotto) |

Esempio:

```bash
python src/cli.py query "Cosa propone il programma sul lavoro?" --category programma --top-k 7 --mode strict
```

### Modalità di risposta

| Modalità | Comportamento | Quando usarla |
|---|---|---|
| `strict` | Risponde solo con le informazioni recuperate. Se non bastano, lo dice. Se nessun chunk supera la soglia, **non chiama nemmeno il modello** | Quando serve fedeltà alla fonte, per esempio riportare cosa dice un documento |
| `standard` | Parte dalle fonti, può integrare con conoscenza generale ma deve **segnalarlo sempre** (es. "Per contesto generale, ...") | Approfondire mantenendo chiaro cosa viene dalla fonte |
| `full` | Usa le fonti come base preferenziale e integra liberamente, senza obbligo di distinguere | Spiegazioni e note personali, **sconsigliata** per riportare fedelmente testi di altri |

Con `standard` e `full`, se la ricerca non trova chunk sopra soglia il modello viene comunque interrogato, senza contesto: `FONTI CONSULTATE` risulterà `[nessuna]`.

### Output

Ogni risposta mostra il testo generato, le fonti consultate e il consumo di token (input, cache, output, reasoning, totale).

## Struttura del progetto

```
personal-ai-knowledge/
├── src/
│   ├── cli.py          # entry point, argomenti, orchestrazione
│   ├── chunking.py     # chunking per parole o per frasi
│   ├── embedding.py    # caricamento modelli, prefissi, generazione embedding
│   ├── storage.py      # SQLite: chunk, log delle query, salvataggio risposte
│   ├── search.py       # similarità coseno e ranking
│   ├── generation.py   # client API, system prompt delle tre modalità
│   └── readers.py      # lettura di .md e .pdf
├── log_data_answer/    # una risposta per file di testo, creata alla prima query
├── .env.example
└── README.md
```

I database `test.db` (chunk) e `logs_v2.db` (log delle query) vengono creati nella cartella da cui si lancia il comando.

## Dettagli tecnici

### Chunking

- **`word`**: taglia ogni `--dimension` parole, con overlap in parole. Semplice, ma può spezzare le frasi.
- **`sentence`** (default): divide il testo in frasi con una regex sulla punteggiatura e le accumula finché si supera `--dimension` parole. La frase che fa superare il tetto viene inclusa per intero, quindi il chunk può eccederlo. L'overlap è il numero di frasi ripetute all'inizio del chunk successivo.

La divisione in frasi è basata su regex: abbreviazioni come "Dott." possono produrre tagli imprecisi.

### Embedding per categoria

| Categoria | Modello | Contesto | Note |
|---|---|---|---|
| `note` | `paraphrase-multilingual-mpnet-base-v2` | 128 token | adatto a prosa breve |
| `programma` | `intfloat/multilingual-e5-large` | 512 token | richiede i prefissi `query:` e `passage:`, applicati automaticamente |

Il limite di token è importante: un modello tronca **in silenzio** il testo oltre il proprio limite, quindi frasi molto lunghe con il modello a 128 token perdono la parte finale. Per questo la categoria `programma` usa il modello a 512 token.

Ogni chunk viene salvato con il nome del modello che ha prodotto il suo embedding. In ricerca vengono letti solo i chunk dello stesso modello della domanda, perché embedding di modelli diversi non sono confrontabili, nemmeno a parità di dimensione del vettore. Le categorie vanno quindi usate in modo coerente tra `index` e `query`.

### Retrieval

Similarità coseno tra domanda e chunk della categoria, ordinamento decrescente, scarto dei risultati sotto `--min-sim`, primi `--top-k`. Il confronto è calcolato in Python su tutti i chunk della categoria.

### Generazione

I chunk recuperati vengono passati al modello dentro un blocco `<context>`, insieme a un system prompt diverso per ogni modalità. La thinking mode del provider è disattivata. Il modello non ha accesso a file, database o web: vede solo quei frammenti.

### Log

Ogni query viene registrata in `logs_v2.db`, tabella `query_log`: modello di embedding, timestamp (UTC), domanda, categoria, numero di risultati, fonti, modalità, modello API, token (input, input in cache, output, reasoning, totale) e tempo di esecuzione. Il tempo misura ricerca e generazione, non il caricamento del modello di embedding.

Il log è in un database separato da quello dei chunk, così si può ricreare `test.db` per sperimentare sull'indicizzazione senza perdere lo storico. Ogni risposta viene salvata anche in `log_data_answer/`, con domanda, risposta, fonti, modalità e categoria.

## Privacy

Indicizzazione e ricerca avvengono in locale, ma per generare la risposta il testo dei chunk recuperati e la domanda vengono **inviati al provider API**. Non indicizzare con questo strumento contenuti che non vuoi condividere con il provider, oppure usa un modello locale al posto dell'API.

## Limiti noti

- **Le soglie di similarità non sono confrontabili tra modelli.** Con `multilingual-e5-large` i punteggi si concentrano in una fascia alta anche per testi scorrelati: una domanda fuori tema ha superato la soglia di default e ha richiesto `--min-sim 0.9` per non recuperare chunk. Il default `0.4` è tarato sulla categoria `note` e con `programma` filtra poco.
- **In generale la similarità coseno non separa nettamente** contenuti rilevanti e irrilevanti; la soglia è un filtro grezzo.
- **`standard` e `full` possono trattare chunk poco pertinenti come fonti valide**, e le fonti elencate non garantiscono che ogni affermazione della risposta derivi da esse. In `full` testo e interpretazione del modello possono fondersi.
- **Risposte non deterministiche**: la temperatura non è impostata, quindi la stessa domanda può dare risposte diverse.
- Nessuna memoria tra le domande: ogni query è indipendente.
- I PDF scansionati (immagini) non sono supportati; l'estrazione del testo da PDF può produrre interruzioni di riga irregolari.
- Con molte migliaia di chunk la ricerca in Python puro rallenterà.

## Roadmap

**Completato**
- [x] Chunking per parole e per frasi
- [x] Storage SQLite e retrieval con soglia minima
- [x] Generazione con elenco delle fonti
- [x] Supporto PDF
- [x] Embedding multi-modello per categoria
- [x] Tre modalità di risposta (`strict`, `standard`, `full`)
- [x] Log strutturato delle query e salvataggio delle risposte

**Prossimo step: soglia di similarità per categoria**
- [ ] Flag `--show-chunks` per vedere testo e punteggio dei chunk recuperati (utile anche per verificare la fedeltà delle risposte)
- [ ] Calibrazione su domande pertinenti e fuori tema per ogni categoria
- [ ] `min_sim` di default legato alla categoria, con `--min-sim` come override
- [ ] Se i punteggi si sovrappongono: soglia relativa al miglior risultato, o re-ranking

**Secondari**
- [ ] Costo cumulativo delle query
- [ ] Modalità interattiva (REPL) per non ricaricare il modello a ogni comando
- [ ] `sqlite-vec` per la ricerca vettoriale nel database (se il volume lo giustifica)
- [ ] Watcher sulla cartella indicizzata
- [ ] Cronologia conversazionale e streaming della risposta
- [ ] Re-ranking e multi-query
- [ ] Sync remoto
