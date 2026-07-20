# ============================================================
#
# Handles persistent conversation history using SQLite.
# SQLite is a lightweight database built into Python.
# No separate database server needed.
#
# What we store:
# - Every question the user asked
# - Every answer the system gave
# - Which document was being queried
# - Timestamp of each exchange
# ============================================================

import sqlite3
import os
import uuid
from datetime import datetime


# Path to SQLite database file
SQLITE_PATH = os.getenv("SQLITE_DB_PATH", "./data/conversation_history.db")


class ConversationMemory:
    """
    Manages persistent conversation history.

    Think of this as a diary that records every conversation
    with the Q&A bot — permanently, across sessions.
    """

    def __init__(self, db_path=SQLITE_PATH):
        """
        Initialize the memory system.
        Creates the database and tables if they don't exist.
        """
        self.db_path = db_path

        # Make sure the directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize the database
        self._create_tables()

    def _get_connection(self):
        """
        Get a database connection.

        We create a new connection each time instead of keeping
        one open permanently. This is safer for multi-user apps.
        check_same_thread=False: Allow connections from different threads
        (Streamlit uses multiple threads)
        """
        return sqlite3.connect(
            self.db_path,
            check_same_thread=False
        )

    def _create_tables(self):
        """
        Create database tables if they don't exist.

        We have two tables:
        1. sessions: Tracks each document session
        2. conversations: Stores Q&A pairs
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Table 1: Sessions (one per document upload)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id   TEXT PRIMARY KEY,
                document_name TEXT NOT NULL,
                document_pages INTEGER,
                document_chunks INTEGER,
                started_at   TEXT NOT NULL,
                last_active  TEXT NOT NULL
            )
        """)

        # Table 2: Conversations (Q&A pairs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                question     TEXT NOT NULL,
                answer       TEXT NOT NULL,
                source_chunks TEXT,
                asked_at     TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Create index for faster queries by session
        # An index is like the index in a book — faster lookup
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session
            ON conversations(session_id)
        """)

        conn.commit()
        conn.close()

    def create_session(self, document_name, pages=0, chunks=0):
        """
        Create a new session when a document is uploaded.
        Returns the unique session ID.
        """
        session_id = str(uuid.uuid4())
        # uuid4(): Generate a random unique ID
        # Example: "550e8400-e29b-41d4-a716-446655440000"

        now = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sessions
            (session_id, document_name, document_pages,
             document_chunks, started_at, last_active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, document_name, pages, chunks, now, now))

        conn.commit()
        conn.close()

        return session_id

    def save_qa_pair(self, session_id, question, answer, source_chunks=None):
        """
        Save a question-answer pair to the database.

        source_chunks: List of text chunks the answer was based on.
                      We store these so users can verify answers.
        """
        import json

        # Convert source chunks list to JSON string for storage
        sources_json = json.dumps(source_chunks) if source_chunks else "[]"
        now = datetime.now().isoformat()

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO conversations
            (session_id, question, answer, source_chunks, asked_at)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, question, answer, sources_json, now))

        # Update session last_active timestamp
        cursor.execute("""
            UPDATE sessions
            SET last_active = ?
            WHERE session_id = ?
        """, (now, session_id))

        conn.commit()
        conn.close()

    def get_session_history(self, session_id):
        """
        Get all Q&A pairs for a specific session.
        Returns list of dictionaries, newest first.
        """
        import json

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT question, answer, source_chunks, asked_at
            FROM conversations
            WHERE session_id = ?
            ORDER BY asked_at ASC
        """, (session_id,))

        rows = cursor.fetchall()
        conn.close()

        # Convert rows to list of dictionaries
        history = []
        for row in rows:
            history.append({
                "question": row[0],
                "answer": row[1],
                "sources": json.loads(row[2]),
                "timestamp": row[3]
            })

        return history

    def get_all_sessions(self):
        """
        Get list of all past sessions.
        Used to show conversation history in the sidebar.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_id, document_name, started_at,
                   last_active, document_pages, document_chunks
            FROM sessions
            ORDER BY last_active DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append({
                "session_id": row[0],
                "document_name": row[1],
                "started_at": row[2],
                "last_active": row[3],
                "pages": row[4],
                "chunks": row[5]
            })

        return sessions

    def get_stats(self):
        """
        Get overall system statistics.
        Shown in the app's about section.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_questions = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(DISTINCT document_name)
            FROM sessions
        """)
        unique_documents = cursor.fetchone()[0]

        conn.close()

        return {
            "total_sessions": total_sessions,
            "total_questions": total_questions,
            "unique_documents": unique_documents
        }