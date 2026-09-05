# Personal AI Knowledge

Sistema RAG (Retrieval-Augmented Generation) locale e offline-first per indicizzare note personali e documenti, e interrogarli tramite un assistente AI che risponde basandosi esclusivamente sui contenuti indicizzati.

## Indice

- [Personal AI Knowledge](#personal-ai-knowledge)
  - [Indice](#indice)
  - [Caratteristiche](#caratteristiche)
  - [Architettura](#architettura)
  - [Requisiti](#requisiti)
  - [Installazione](#installazione)
  - [Configurazione](#configurazione)
  - [Utilizzo](#utilizzo)
    - [Indicizzare documenti](#indicizzare-documenti)
    - [Fare domande](#fare-domande)
  - [Struttura del progetto](#struttura-del-progetto)
  - [Dettagli tecnici](#dettagli-tecnici)
    - [Chunking](#chunking)
    - [Embedding multi-modello](#embedding-multi-modello)
    - [Retrieval](#retrieval)
    - [Generazione](#generazione)
    - [Logging](#logging)
  - [Roadmap](#roadmap)
  - [Limiti noti](#limiti-noti)

## Caratteristiche

- Indicizzazione di file **Markdown (.md)** e **PDF (.pdf)**
- Chunking intelligente per frasi o parole con overlap configurabile
- Embedding **multilingua**, con supporto a modelli diversi per categorie di contenuto diverse (es. note personali vs documenti più lunghi)
- Storage locale su **SQLite**, nessuna dipendenza da servizi cloud per l'indicizzazione
- Retrieval semantico con soglia minima di similarità configurabile
- Generazione delle risposte tramite API compatibile OpenAI (attualmente **DeepSeek**), con modalità che regolano quanto il modello può discostarsi dalle fonti indicizzate
- Citazione delle fonti per ogni risposta
- Logging strutturato di ogni query (tempo di esecuzione, token usati, fonti) e salvataggio testuale delle risposte

## Architettura

Il flusso di lavoro si divide in due fasi principali:

**Indicizzazione** (`index`): i file vengono letti, divisi in chunk, trasformati in embedding e salvati nel database.

```
File (.md/.pdf) → lettura → chunking → embedding → storage (SQLite)
```

**Interrogazione** (`query`): la domanda viene trasformata in embedding, confrontata con i chunk salvati, e i risultati più pertinenti vengono passati come contesto a un modello linguistico che genera la risposta.

```
Domanda → embedding → ricerca semantica (top-k) → generazione risposta (LLM) → output + log
```

## Requisiti

- Python 3.12+
- Un account [DeepSeek](https://platform.deepseek.com) con API key
- Consigliata una GPU per l'indicizzazione (su CPU la generazione degli embedding, specialmente con il modello per la categoria `programma`, può essere molto lenta ad esempio su Google Colab con GPU T4 l'indicizzazione di ~300 chunk richiede pochi secondi contro diversi minuti su CPU)

## Installazione

```bash
git clone https://github.com/ZhoupengWu/personal-ai-knowledge.git
cd personal-ai-knowledge

python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configurazione

Crea un file `.env` nella cartella del progetto con:

```
DEEPSEEK_API_KEY=la-tua-api-key
DEEPSEEK_MODEL=deepseek-v4-flash
```

`DEEPSEEK_MODEL` è opzionale (default: `deepseek-v4-flash`).

## Utilizzo

### Indicizzare documenti

```bash
python src/cli.py index <cartella> [opzioni]
```

| Opzione | Default | Descrizione |
|---|---|---|
| `folder` |  | Cartella contenente i file `.md`/`.pdf` da indicizzare (obbligatorio) |
| `--strategy` | `sentence` | `sentence` raggruppa frasi intere fino al limite di parole; `word` taglia per numero fisso di parole (può spezzare le frasi) |
| `--dimension` | `40` | Numero massimo di parole per chunk (vedi [Chunking](#chunking) per il comportamento esatto in base alla strategia) |
| `--overlap` | `1` | Sovrapposizione tra chunk consecutivi (in frasi con `sentence`, in parole con `word`) |
| `--category` | `note` | Categoria del contenuto: determina quale modello di embedding viene usato (vedi [Embedding multi-modello](#embedding-multi-modello)) |

Esempio:
```bash
python src/cli.py index ./documenti/programma-politico --category programma --dimension 40 --overlap 1
```

Re-indicizzare una cartella già indicizzata è sicuro: i chunk associati a ciascun file vengono cancellati e ricreati, senza duplicati.

### Fare domande

```bash
python src/cli.py query "<domanda>" [opzioni]
```

| Opzione | Default | Descrizione |
|---|---|---|
| `text` |  | La domanda da porre (obbligatorio) |
| `--category` | `note` | Categoria su cui cercare — deve corrispondere alla categoria usata in fase di indicizzazione |
| `--top-k` | `5` | Numero massimo di chunk passati come contesto al modello |
| `--min-sim` | `0.4` | Soglia minima di similarità sotto la quale un chunk viene scartato (filtro di sicurezza contro il rumore, non un filtro di rilevanza fine) |
| `--mode` | `strict` | Quanto il modello può discostarsi dalle fonti indicizzate (vedi [Generazione](#generazione)) |

Esempio:
```bash
python src/cli.py query "Cosa propone il programma sul tema del lavoro?" --category programma --top-k 8
```

Ogni risposta include: il testo generato, le fonti utilizzate, e i token consumati (input, cache, output, reasoning, totale).

## Struttura del progetto

```
personal-ai-knowledge/
├── src/
│   ├── cli.py            # entry point, parsing argomenti, orchestrazione
│   ├── chunking.py       # divisione del testo in chunk (per parole o per frasi)
│   ├── embedding.py      # caricamento modelli e generazione embedding
│   ├── storage.py        # persistenza SQLite (chunk, log delle query)
│   ├── search.py         # similarità coseno e ranking dei risultati
│   ├── generation.py     # chiamata all'API di generazione e system prompt
│   └── readers.py        # lettura dei file sorgente (.md, .pdf)
├── .env                  # configurazione locale
├── .env.example          # esempio configurazione locale
├── .gitignore
├── requirements.txt
└── README.md
```

## Dettagli tecnici

### Chunking

Due strategie disponibili:

- **`word`**: taglia il testo ogni N parole esatte, con overlap in parole. Semplice e prevedibile, ma può spezzare frasi a metà.
- **`sentence`** (default): divide il testo in frasi (tramite regex su punteggiatura), poi raggruppa frasi intere finché non si supera il tetto di `--dimension` parole. L'ultima frase che fa scattare la soglia viene sempre inclusa per intero quindi il chunk risultante può superare leggermente `--dimension`. L'overlap è espresso in numero di frasi ripetute tra un chunk e il successivo, non in parole.

### Embedding multi-modello

Categorie diverse di contenuto usano modelli di embedding diversi, per adattarsi a caratteristiche testuali diverse (es. lunghezza media delle frasi):

| Categoria | Modello | Note |
|---|---|---|
| `note` | `paraphrase-multilingual-mpnet-base-v2` | Contesto di 128 token, adatto a prosa breve |
| `programma` | `intfloat/multilingual-e5-large` | Contesto di 512 token, richiede prefissi obbligatori `query:`/`passage:` applicati automaticamente |

Ogni chunk viene salvato insieme al nome del modello che ha generato il suo embedding. In fase di ricerca, i chunk vengono filtrati per modello compatibile con quello usato per la domanda quindi embedding generati da modelli diversi non sono comparabili tra loro con la similarità coseno, anche a parità di dimensione del vettore.

### Retrieval

La ricerca calcola la similarità coseno tra l'embedding della domanda e ogni chunk della categoria selezionata, ordina per punteggio decrescente, scarta i risultati sotto `--min-sim`, e restituisce i primi `--top-k`.

### Generazione

Le risposte sono generate da un modello LLM (DeepSeek) a cui vengono passati i chunk recuperati come contesto. Il comportamento è regolato dal parametro `--mode`:

- **`strict`** (default): risponde esclusivamente sulla base dei chunk forniti. Se l'informazione non è presente, lo dichiara esplicitamente senza proporre alternative o integrazioni con conoscenza esterna.
- **`hybrid`**: non ancora implementata, ricade automaticamente su `strict`.

Il modello non ha accesso diretto ai file, al database o al web: riceve esclusivamente il testo dei chunk selezionati dal retrieval per quella specifica domanda.

### Logging

Ogni query viene registrata in un database separato (`logs.db`), distinto dal database dei chunk (`test.db`) per poter sperimentare liberamente con parametri di indicizzazione diversi senza perdere lo storico delle interrogazioni. Vengono salvati: timestamp (UTC), domanda, categoria, fonti, numero di risultati, token consumati (con dettaglio cache/reasoning) e tempo di esecuzione.

Ogni risposta viene inoltre salvata come file di testo individuale in `log_data_answer/`, con nome basato sul timestamp della query.

## Roadmap

**Completati**
- [x] Chunking per parole e per frasi
- [x] Storage SQLite con embedding come BLOB
- [x] Retrieval con soglia minima di similarità
- [x] Generazione con citazione delle fonti
- [x] Supporto PDF
- [x] Embedding multi-modello per categoria
- [x] Logging strutturato delle query e salvataggio risposte su file

**Da fare**
- [ ] Modalità di generazione aggiuntive (oltre a `strict`/`hybrid`)
- [ ] `sqlite-vec` per ricerca vettoriale nativa (da valutare se il volume di chunk lo giustifica)
- [ ] Costo cumulativo delle query
- [ ] Modalità interattiva (REPL) per non ricaricare il modello ad ogni comando
- [ ] Watcher automatico sulla cartella indicizzata
- [ ] Cronologia conversazionale (domande di follow-up)
- [ ] Streaming della risposta
- [ ] Re-ranking e multi-query
- [ ] Sync remoto / multi-dispositivo

## Limiti noti

- La similarità coseno con i modelli usati non separa nettamente contenuti rilevanti da irrilevanti: raramente scende sotto 0.3–0.4 anche per testi scorrelati. La soglia `--min-sim` va quindi intesa come filtro grezzo contro il rumore più evidente, non come garanzia di rilevanza.
- La ricerca carica ed elabora in Python i chunk della categoria selezionata: con volumi molto grandi (decine di migliaia di chunk) le performance potrebbero degradare.
- Nessuna gestione di conversazioni multi-turno: ogni query è indipendente, senza memoria delle domande precedenti.