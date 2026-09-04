import sqlite3
import numpy as np
from pathlib import Path

def createConnection(path: str):
    return sqlite3.connect(path)

def createTableChunk(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            embedding BLOB,
            source TEXT,
            hf_model_name TEXT
        )
    """)

    cur.close()
    conn.commit()

def insertChunk(conn: sqlite3.Connection, text: str, embedding: np.ndarray, source: str, hf_model_name: str) -> None:
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO chunks (text, embedding, source, hf_model_name)
        VALUES (?, ?, ?, ?)
    """, (text, embedding.tobytes(), source, hf_model_name))

    cur.close()
    conn.commit()

def getAllChunks(conn: sqlite3.Connection) -> list[tuple] | None:
    cur = conn.cursor()

    cur.execute("""
        SELECT text, embedding, source, hf_model_name
        FROM chunks
    """)

    data = cur.fetchall()

    if not data:
        return None

    new_data = []

    for i in range(len(data)):
        blob = np.frombuffer(data[i][1], dtype=np.float32)
        new_data.append((data[i][0], blob, data[i][2], data[i][3]))

    return new_data

def getChunksByModel(conn: sqlite3.Connection, hf_model_name: str) -> list[tuple] | None:
    cur = conn.cursor()

    cur.execute("""
        SELECT text, embedding, source, hf_model_name
        FROM chunks
        WHERE hf_model_name = ?
    """, (hf_model_name,))

    data = cur.fetchall()

    if not data:
        return None

    new_data = []

    for i in range(len(data)):
        blob = np.frombuffer(data[i][1], dtype=np.float32)
        new_data.append((data[i][0], blob, data[i][2], data[i][3]))

    return new_data

def deleteChunksBySource(conn: sqlite3.Connection, source: str) -> None:
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM chunks
        WHERE source = ?
    """, (source,))

    cur.close()
    conn.commit()

def createTableLog(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_hf TEXT,
            timestamp TEXT,
            query TEXT,
            category TEXT,
            num_results INTEGER,
            sources TEXT,
            model_name_api TEXT,
            input_tokens INTEGER,
            input_cached_tokens INTEGER,
            output_tokens INTEGER,
            reasoning_tokens INTEGER,
            total_tokens INTEGER,
            elapsed_seconds REAL
        )
    """)

    cur.close()
    conn.commit()

def logQuery(conn: sqlite3.Connection, model_hf: str, timestamp: str, query: str, category: str, num_results: int, sources: str, model_name_api: str, inp_tokens: int, inp_cached_tokens: int, out_tokens: int, reasoning_tokens: int, total_tokens: int, elapsed_seconds: float) -> None:
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO query_log (model_hf, timestamp, query, category, num_results, sources, model_name_api, input_tokens, input_cached_tokens, output_tokens, reasoning_tokens, total_tokens, elapsed_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (model_hf, timestamp, query, category, num_results, sources, model_name_api, inp_tokens, inp_cached_tokens, out_tokens, reasoning_tokens, total_tokens, elapsed_seconds))

    cur.close()
    conn.commit()

def saveAnswerToFile(query: str, answer: str, timestamp: str, sources: str) -> None:
    safe_timestamp = timestamp.replace(":", "_")
    filename = f"query_{safe_timestamp}.txt"

    folder = Path("log_data_answer")
    folder.mkdir(exist_ok=True)

    file_path = folder / filename
    file_path.write_text(f"Domanda: {query}\n\nRisposta:\n{answer}\n\n[{sources}]", encoding="utf-8")
