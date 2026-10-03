# Roadmap

Stato del progetto e prossimi passi. Per cosa fa oggi il sistema e come si usa vedi il [README](README.md).

## Completato

- [x] Chunking per parole e per frasi
- [x] Storage SQLite e retrieval con soglia minima
- [x] Generazione con elenco delle fonti
- [x] Supporto PDF
- [x] Embedding multi-modello per categoria
- [x] Tre modalità di risposta (`strict`, `standard`, `full`)
- [x] Log strutturato delle query e salvataggio delle risposte
- [x] Flag `--show-chunks` per vedere testo e punteggio dei chunk recuperati
- [x] Calibrazione su domande pertinenti e fuori tema per ogni categoria
- [x] `min_sim` di default legato alla categoria, con `--min-sim` come override
- [x] Temperatura legata alla modalità, con `--temperature` come override
- [x] Comando `eval` per calibrare le soglie su domande etichettate

## Prossimo step

- [ ] Re-ranking dei chunk recuperati (la soglia da sola non separa i casi vicini per tema o i falsi positivi in `note`)

## Secondari

- [ ] Costo cumulativo delle query
- [ ] Modalità interattiva (REPL) per non ricaricare il modello a ogni comando
- [ ] `sqlite-vec` per la ricerca vettoriale nel database (se il volume lo giustifica)
- [ ] Watcher sulla cartella indicizzata
- [ ] Cronologia conversazionale e streaming della risposta
- [ ] Multi-query
- [ ] Sync remoto
