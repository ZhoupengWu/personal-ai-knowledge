import json
from embedding import embeddedTexts
from search import search

LABELS = ("pertinente", "borderline", "fuori_tema")
REQUIRED_FIELDS = ("question", "category", "label")

def loadQuestions(path: str, valid_categories: list[str]) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)

    for i, q in enumerate(questions, start=1):
        for field in REQUIRED_FIELDS:
            if field not in q:
                raise ValueError(f"Domanda {i}: manca il campo '{field}'")

        if q["label"] not in LABELS:
            raise ValueError(f"Domanda {i}: label '{q['label']}' non valida (usa: {', '.join(LABELS)})")

        if q["category"] not in valid_categories:
            raise ValueError(f"Domanda {i}: categoria '{q['category']}' sconosciuta (usa: {', '.join(valid_categories)})")

        if isinstance(q.get("expected_source"), str):
            q["expected_source"] = [q["expected_source"]]

    return questions

def groupByCategory(questions: list[dict]) -> dict[str, list[dict]]:
    groups = {}

    for q in questions:
        groups.setdefault(q["category"], []).append(q)

    return groups

def scoreQuestions(model, hf_model_name: str, chunks: list[tuple], questions: list[dict], top_k: int) -> list[dict]:
    embeddings = embeddedTexts(model, [q["question"] for q in questions], hf_model_name, "query")
    results = []

    for q, embedding in zip(questions, embeddings):
        found = search(embedding, chunks, top_k, -1.0)

        results.append({
            "question": q["question"],
            "label": q["label"],
            "expected_source": q.get("expected_source"),
            "scores": [float(chunk[1]) for chunk in found],
            "sources": [chunk[2] for chunk in found]
        })

    return results

def summarize(results: list[dict], threshold: float) -> dict:
    relevant = [r for r in results if r["label"] == "pertinente" and r["scores"]]
    off_topic = [r for r in results if r["label"] == "fuori_tema" and r["scores"]]
    summary = {"threshold": threshold, "n_relevant": len(relevant), "n_off_topic": len(off_topic)}

    if relevant:
        summary["relevant_top1_min"] = min(r["scores"][0] for r in relevant)
        summary["relevant_all_min"] = min(min(r["scores"]) for r in relevant)
        summary["answers_lost"] = [r["question"] for r in relevant if r["scores"][0] < threshold]
        summary["chunks_cut"] = sum(1 for r in relevant for s in r["scores"] if s < threshold)
        summary["chunks_total"] = sum(len(r["scores"]) for r in relevant)
        summary["wrong_source"] = [
            r["question"] for r in relevant
            if r["expected_source"] and r["sources"][0] not in r["expected_source"]
        ]

    if off_topic:
        summary["off_topic_max"] = max(max(r["scores"]) for r in off_topic)
        summary["leaked"] = [r["question"] for r in off_topic if r["scores"][0] >= threshold]
        summary["leaked_chunks"] = sum(1 for r in off_topic for s in r["scores"] if s >= threshold)

    if relevant and off_topic:
        summary["margin_all"] = summary["relevant_all_min"] - summary["off_topic_max"]
        summary["margin_top1"] = summary["relevant_top1_min"] - summary["off_topic_max"]

        if summary["margin_all"] > 0:
            summary["suggested"] = (summary["off_topic_max"] + summary["relevant_all_min"]) / 2
            summary["suggested_mode"] = "all"
        elif summary["margin_top1"] > 0:
            summary["suggested"] = (summary["off_topic_max"] + summary["relevant_top1_min"]) / 2
            summary["suggested_mode"] = "top1"

    return summary

def sweepThresholds(threshold: float) -> list[float]:
    step = 0.01 if threshold >= 0.7 else 0.05

    return [round(threshold + step * k, 2) for k in range(-2, 5)]

def sweep(results: list[dict], thresholds: list[float]) -> list[dict]:
    relevant = [r for r in results if r["label"] == "pertinente" and r["scores"]]
    off_topic = [r for r in results if r["label"] == "fuori_tema" and r["scores"]]
    rows = []

    for t in thresholds:
        rows.append({
            "threshold": t,
            "answers_lost": sum(1 for r in relevant if r["scores"][0] < t),
            "chunks_cut": sum(1 for r in relevant for s in r["scores"] if s < t),
            "chunks_total": sum(len(r["scores"]) for r in relevant),
            "leaked": sum(1 for r in off_topic if r["scores"][0] >= t),
            "leaked_chunks": sum(1 for r in off_topic for s in r["scores"] if s >= t),
            "n_off_topic": len(off_topic)
        })

    return rows

def printReport(category: str, results: list[dict], summary: dict, verbose: bool = False):
    threshold = summary["threshold"]
    order = {label: i for i, label in enumerate(LABELS)}
    sorted_results = sorted(results, key=lambda r: (order[r["label"]], -r["scores"][0] if r["scores"] else 0))

    print(f"\n=== Categoria: {category} (soglia attuale {threshold}) ===\n")
    print(f"{'etichetta':<11} {'1°':>6} {'ultimo':>7}  {'esito':<10} {'fonte':<6} domanda")

    for r in sorted_results:
        if not r["scores"]:
            print(f"{r['label']:<11} {'-':>6} {'-':>7}  nessun risultato  {r['question']}")
            continue

        top1 = r["scores"][0]
        last = r["scores"][-1]
        outcome = "passa" if top1 >= threshold else "scartata"

        if r["expected_source"]:
            source_check = "ok" if r["sources"][0] in r["expected_source"] else "ERRATA"
        else:
            source_check = "-"

        print(f"{r['label']:<11} {top1:>6.3f} {last:>7.3f}  {outcome:<10} {source_check:<6} {r['question'][:70]}")

        if verbose:
            for score, source in zip(r["scores"], r["sources"]):
                print(f"{'':<12}{score:>6.3f}  {source}")

    print("\n--- Riepilogo ---")

    if "relevant_top1_min" in summary:
        print(f"Pertinenti ({summary['n_relevant']}): primo risultato più basso {summary['relevant_top1_min']:.3f}, "
              f"chunk più basso in assoluto {summary['relevant_all_min']:.3f}")

    if "off_topic_max" in summary:
        print(f"Fuori tema ({summary['n_off_topic']}): chunk più alto {summary['off_topic_max']:.3f}")

    if "margin_all" in summary:
        print(f"Margine tenendo tutti i chunk delle pertinenti: {summary['margin_all']:+.3f}")
        print(f"Margine tenendo solo il primo di ogni pertinente: {summary['margin_top1']:+.3f}")

        if "suggested" in summary:
            detail = "tiene tutti i chunk delle pertinenti" if summary["suggested_mode"] == "all" else "tiene il primo chunk di ogni pertinente, ma ne taglia alcuni successivi"
            print(f"Soglia suggerita: {summary['suggested']:.3f} ({detail})")
        else:
            print("Soglia suggerita: nessuna, i punteggi di pertinenti e fuori tema si sovrappongono")

    if summary["n_relevant"] and summary["n_off_topic"]:
        print("\nSoglie a confronto:")
        print(f"{'soglia':>7}  {'pertinenti senza chunk':>23}  {'chunk pertinenti tagliati':>26}  {'fuori tema che passano':>23}")

        for row in sweep(results, sweepThresholds(threshold)):
            mark = "  <- attuale" if row["threshold"] == threshold else ""
            print(f"{row['threshold']:>7.2f}  {row['answers_lost']:>23}  {str(row['chunks_cut']) + '/' + str(row['chunks_total']):>26}  "
                  f"{str(row['leaked']) + '/' + str(row['n_off_topic']) + ' (' + str(row['leaked_chunks']) + ' chunk)':>23}{mark}")

    print(f"\nCon la soglia attuale ({threshold}):")

    if "answers_lost" in summary:
        print(f"- pertinenti senza nessun chunk: {len(summary['answers_lost'])}/{summary['n_relevant']}")
        print(f"- chunk di pertinenti tagliati: {summary['chunks_cut']}/{summary['chunks_total']}")

        for question in summary["answers_lost"]:
            print(f"    persa: {question}")

        for question in summary["wrong_source"]:
            print(f"    fonte diversa da quella attesa: {question}")

    if "leaked" in summary:
        print(f"- fuori tema con almeno un chunk che passa: {len(summary['leaked'])}/{summary['n_off_topic']} ({summary['leaked_chunks']} chunk)")

        for question in summary["leaked"]:
            print(f"    passa: {question}")

    borderline = [r for r in results if r["label"] == "borderline" and r["scores"]]

    if borderline:
        passing = sum(1 for r in borderline if r["scores"][0] >= threshold)
        print(f"- borderline con almeno un chunk che passa: {passing}/{len(borderline)} (informativo, non entra nel calcolo)")
