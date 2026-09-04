import sqlite3
import numpy as np

def createConnection(path: str):
    return sqlite3.connect(path)

def createTable(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            embedding BLOB,
            source TEXT,
            model_name TEXT
        )
    """)

    cur.close()
    conn.commit()

def insertChunk(conn: sqlite3.Connection, text: str, embedding: np.ndarray, source: str, model_name: str) -> None:
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO chunks (text, embedding, source, model_name)
        VALUES (?, ?, ?, ?)
    """, (text, embedding.tobytes(), source, model_name))

    cur.close()
    conn.commit()

def getAllChunks(conn: sqlite3.Connection) -> list[tuple] | None:
    cur = conn.cursor()

    cur.execute("""
        SELECT text, embedding, source, model_name
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

def deleteChunksBySource(conn: sqlite3.Connection, source: str):
    cur = conn.cursor()

    cur.execute("""
        DELETE FROM chunks
        WHERE source = ?
    """, (source,))

    cur.close()
    conn.commit()
