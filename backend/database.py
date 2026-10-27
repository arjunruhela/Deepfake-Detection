import sqlite3
import os
import hashlib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "provenance.db")


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_table():
    """Create the analysis history table and safely upgrade older databases."""
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            sha256_hash TEXT NOT NULL,
            prediction TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add the new column to existing databases without deleting their records.
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(analysis_history)").fetchall()
    }

    if "perceptual_hash" not in columns:
        connection.execute(
            "ALTER TABLE analysis_history ADD COLUMN perceptual_hash TEXT"
        )

    connection.commit()
    connection.close()


def calculate_sha256(file_bytes):
    """Return the SHA-256 hash of the exact file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def save_analysis(
    filename,
    file_type,
    sha256_hash,
    prediction,
    perceptual_hash=None
):
    """Save an analysis result, optionally including an image perceptual hash."""
    connection = get_connection()

    connection.execute("""
        INSERT INTO analysis_history
        (filename, file_type, sha256_hash, prediction, perceptual_hash)
        VALUES (?, ?, ?, ?, ?)
    """, (filename, file_type, sha256_hash, prediction, perceptual_hash))

    connection.commit()
    connection.close()


def get_history():
    connection = get_connection()

    rows = connection.execute("""
        SELECT id, filename, file_type, sha256_hash,
               prediction, created_at
        FROM analysis_history
        ORDER BY id DESC
    """).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def find_duplicate(sha256_hash):
    """Find the earliest record with the exact same SHA-256 hash."""
    connection = get_connection()

    record = connection.execute("""
        SELECT id, filename, file_type, sha256_hash,
               prediction, created_at
        FROM analysis_history
        WHERE sha256_hash = ?
        ORDER BY id ASC
        LIMIT 1
    """, (sha256_hash,)).fetchone()

    connection.close()

    return dict(record) if record else None


def find_similar_images(perceptual_hash, max_distance=8):
    """
    Find previously analyzed images whose perceptual hashes are close.

    Both hashes must be hexadecimal strings of the same length. The Hamming
    distance counts the differing bits. A smaller distance means the hashes
    are more alike; max_distance=8 is a starting threshold, not a guarantee
    that two files depict the same original image.
    """
    if not perceptual_hash:
        return []

    try:
        target_value = int(perceptual_hash, 16)
    except (TypeError, ValueError):
        raise ValueError("perceptual_hash must be a hexadecimal string")

    connection = get_connection()

    rows = connection.execute("""
        SELECT id, filename, file_type, sha256_hash,
               prediction, created_at, perceptual_hash
        FROM analysis_history
        WHERE file_type = 'image'
          AND perceptual_hash IS NOT NULL
          AND LENGTH(perceptual_hash) = ?
        ORDER BY id ASC
    """, (len(perceptual_hash),)).fetchall()

    connection.close()

    matches = []
    for row in rows:
        record = dict(row)
        try:
            previous_value = int(record["perceptual_hash"], 16)
        except (TypeError, ValueError):
            continue

        distance = (target_value ^ previous_value).bit_count()

        if distance <= max_distance:
            record["hash_distance"] = distance
            record.pop("perceptual_hash", None)
            matches.append(record)

    return sorted(matches, key=lambda item: item["hash_distance"])


if __name__ == "__main__":
    create_table()
    print("Database is ready!")
    print(f"Database path: {DATABASE_PATH}")
