# ============================================================
#
# Manages the ChromaDB vector store.
# Handles: storing chunks, searching for similar chunks,
# managing multiple document collections.
# ============================================================

import os
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

# Suppress ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# ChromaDB storage path from environment variable
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

# Embedding model name
# This MUST be the same model used during indexing
# If you index with model A and search with model B
# the vectors are incompatible and results will be wrong
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# How many chunks to retrieve per query
# 4 is optimal for our use case:
# Too few (1-2): May miss context
# Too many (8+): Overloads LLM context window, adds noise
DEFAULT_TOP_K = 4


class VectorStoreManager:
    """
    Manages all interactions with ChromaDB vector storage.

    Responsibilities:
    - Loading the embedding model
    - Storing document chunks as vectors
    - Searching for relevant chunks by semantic similarity
    - Managing multiple collections (one per document)
    """

    def __init__(self):
        """
        Initialize ChromaDB client and embedding model.
        This is called once when the app starts.
        """
        print("Initializing Vector Store Manager...")

        # Connect to persistent ChromaDB storage
        # PersistentClient: All data saved to disk automatically
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        print(f"   ChromaDB connected: {CHROMA_PATH}")

        # Load embedding model
        # This model runs locally — no API call, no cost
        # First run downloads the model (~90MB), cached after that
        print("   Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"   Embedding model loaded: {EMBEDDING_MODEL}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of text strings into embedding vectors.

        Input:  ["stuck pipe at 1654 feet", "mud weight 10.2 ppg"]
        Output: [[0.23, -0.45, ...], [0.11, 0.87, ...]]
                (list of 384-number vectors)

        We use batch encoding for efficiency —
        encoding 100 texts at once is much faster than
        encoding them one by one in a loop.
        """
        # encode(): Convert texts to numpy array of vectors
        # show_progress_bar=False: No progress output (cleaner logs)
        # batch_size=32: Process 32 texts at a time internally
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=False,
            batch_size=32,
            normalize_embeddings=True
            # normalize_embeddings=True: Makes cosine similarity
            # equivalent to dot product — slightly faster search
        )

        # Convert numpy array to Python list (ChromaDB requires lists)
        return embeddings.tolist()

    def get_or_create_collection(self, collection_name: str):
        """
        Get an existing ChromaDB collection or create a new one.

        A collection = all indexed chunks from one document.
        Collection name is derived from the document filename.

        ChromaDB collection name rules:
        - Must be 3-63 characters
        - Only alphanumeric and underscores
        - Must start and end with alphanumeric
        """
        # Sanitize name to meet ChromaDB requirements
        safe_name = self._sanitize_collection_name(collection_name)

        collection = self.client.get_or_create_collection(
            name=safe_name,
            # metadata: Store info about this collection
            metadata={"document": collection_name}
        )

        return collection

    def _sanitize_collection_name(self, name: str) -> str:
        """
        Convert any filename to a valid ChromaDB collection name.

        Examples:
        "DDR_Day04_2024-01-18.txt" → "ddr_day04_2024_01_18"
        "Discovery_report.pdf"     → "discovery_report"
        "Volve PUD .pdf"           → "volve_pud"
        """
        import re

        # Remove file extension
        name = re.sub(r'\.[a-zA-Z]+$', '', name)

        # Replace spaces, hyphens, dots with underscores
        name = re.sub(r'[\s\-\.]+', '_', name)

        # Remove any remaining non-alphanumeric characters
        name = re.sub(r'[^a-zA-Z0-9_]', '', name)

        # Convert to lowercase
        name = name.lower()

        # Ensure it starts with a letter (ChromaDB requirement)
        if name and name[0].isdigit():
            name = 'doc_' + name

        # Truncate to 63 characters (ChromaDB limit)
        name = name[:63]

        # Ensure minimum length
        if len(name) < 3:
            name = 'doc_' + name

        return name

    def add_chunks(self, chunks: List[Dict],
                   collection_name: str) -> int:
        """
        Add document chunks to ChromaDB.

        Each chunk becomes a searchable entry in the vector store.
        We compute embeddings here and store them alongside the text.

        chunks: List of chunk dicts from SmartChunker
        collection_name: Usually the document filename

        Returns: Number of chunks successfully added
        """
        if not chunks:
            return 0

        collection = self.get_or_create_collection(collection_name)

        # Check which chunks are already in the collection
        # (Prevents duplicate indexing if we run twice)
        existing_ids = set()
        try:
            existing = collection.get(include=[])
            existing_ids = set(existing['ids'])
        except Exception:
            pass  # Empty collection — no existing IDs

        # Filter out chunks that are already indexed
        new_chunks = [
            chunk for chunk in chunks
            if chunk['id'] not in existing_ids
        ]

        if not new_chunks:
            print(f"    All {len(chunks)} chunks already indexed")
            return 0

        # Extract texts for embedding
        texts = [chunk['text'] for chunk in new_chunks]
        ids = [chunk['id'] for chunk in new_chunks]
        metadatas = [
            {
                'document': chunk['document'],
                'chunk_index': chunk['chunk_index'],
                'char_count': chunk['char_count']
            }
            for chunk in new_chunks
        ]

        # Create embeddings for all new chunks
        print(f"    Creating embeddings for {len(new_chunks)} chunks...")
        embeddings = self.embed_texts(texts)

        # Add to ChromaDB in batches of 100
        # (Prevents memory issues with very large documents)
        batch_size = 100
        added = 0

        for i in range(0, len(new_chunks), batch_size):
            batch_end = min(i + batch_size, len(new_chunks))

            collection.add(
                documents=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                ids=ids[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            added += batch_end - i

        print(f"     Added {added} new chunks to '{collection_name}'")
        return added

    def search(self, query: str, collection_name: str,
               top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """
        Search a specific document collection for relevant chunks.

        This is the RETRIEVAL step in RAG:
        1. Convert query to vector
        2. Find most similar vectors in ChromaDB
        3. Return the corresponding text chunks

        Returns list of result dicts, each containing:
        - text: The chunk content
        - document: Source document name
        - score: Similarity score (lower distance = more relevant)
        - chunk_index: Position in original document
        """
        # Convert query to vector using same model as indexing
        query_embedding = self.embed_texts([query])[0]

        try:
            collection = self.client.get_collection(
                self._sanitize_collection_name(collection_name)
            )

            # Query ChromaDB for most similar chunks
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count()),
                # include: What data to return with results
                include=['documents', 'metadatas', 'distances']
            )

            # Format results into clean list of dicts
            formatted = []
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted.append({
                    'text': doc,
                    'document': meta.get('document', collection_name),
                    'score': round(dist, 4),
                    # Convert distance to similarity percentage
                    # Distance 0 = identical, Distance 2 = completely different
                    'relevance_pct': round((1 - dist/2) * 100, 1),
                    'chunk_index': meta.get('chunk_index', i)
                })

            return formatted

        except Exception as e:
            print(f"      Search error in '{collection_name}': {e}")
            return []

    def search_all_collections(self, query: str,
                               top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """
        Search ALL indexed documents simultaneously.

        This is the key feature that makes our system powerful:
        when a user asks a question, we search every document
        and return the most relevant chunks from all of them.

        Process:
        1. Get list of all collections
        2. Search each collection for the query
        3. Combine all results
        4. Sort by relevance (best match first)
        5. Return top_k overall results
        """
        # Get all collections
        all_collections = self.client.list_collections()

        if not all_collections:
            return []

        all_results = []

        # Search each collection
        for collection_info in all_collections:
            results = self.search(
                query=query,
                collection_name=collection_info.name,
                top_k=2  # Get top 2 from each collection
            )
            all_results.extend(results)

        # Sort all results by score (lower distance = better)
        all_results.sort(key=lambda x: x['score'])

        # Return top_k overall best results
        return all_results[:top_k]

    def list_indexed_documents(self) -> List[Dict]:
        """
        List all documents currently indexed in ChromaDB.
        Used in the Streamlit UI to show what's available.
        """
        collections = self.client.list_collections()
        docs = []

        for col_info in collections:
            try:
                collection = self.client.get_collection(col_info.name)
                docs.append({
                    'collection_name': col_info.name,
                    'chunk_count': collection.count()
                })
            except Exception:
                pass

        return docs

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a document has already been indexed.
        Used to skip re-indexing of already-processed documents.
        """
        safe_name = self._sanitize_collection_name(collection_name)
        existing = self.client.list_collections()
        existing_names = [c.name for c in existing]
        return safe_name in existing_names