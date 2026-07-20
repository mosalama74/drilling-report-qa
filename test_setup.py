# ============================================================
# test_setup.py — Full Setup Verification
# 
# Tests ALL components including:
# - OCR (Tesseract + pdf2image + pytesseract)
# - ChromaDB (persistent vector database)
# - SQLite (conversation memory)
# - LangChain, Streamlit, HuggingFace
#
# Run with: python test_setup.py
# ============================================================

import sys
import os

# Suppress ChromaDB telemetry warnings
# ChromaDB tries to send anonymous usage data — this disables that
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

# Load environment variables from .env file before any tests
from dotenv import load_dotenv
load_dotenv()
# After this line, os.getenv("HUGGINGFACEHUB_API_TOKEN") will work


# ============================================================
# TEST 1: Python Version
# ============================================================
def test_python_version():
    """
    Check Python is 3.10 or higher.
    Some libraries we use require modern Python features.
    """
    version = sys.version_info
    assert version.major == 3 and version.minor >= 10, \
        f"Need Python 3.10+, you have {version.major}.{version.minor}"
    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")


# ============================================================
# TEST 2: pdfplumber (Digital PDF extraction)
# ============================================================
def test_pdfplumber():
    """
    Check pdfplumber is installed correctly.
    Used for extracting text from digital (non-scanned) PDFs.
    """
    import pdfplumber
    print(f"  ✅ pdfplumber {pdfplumber.__version__}")


# ============================================================
# TEST 3: Tesseract OCR Engine (System-level installation)
# ============================================================
def test_tesseract_engine():
    """
    Check Tesseract OCR engine is installed on the system.
    Updated to handle different Windows output formats.
    """
    import subprocess

    try:
        result = subprocess.run(
            ['tesseract', '--version'],
            capture_output=True,
            text=True
        )

        # Tesseract sometimes prints to stderr, sometimes stdout
        # Check both and use whichever has content
        output = result.stderr.strip() or result.stdout.strip()

        if output:
            # Get just the first line (version line)
            version_line = output.split('\n')[0].strip()
            print(f"  ✅ Tesseract installed: {version_line}")
        else:
            # If output is empty but command succeeded (return code 0)
            # it means Tesseract is installed but version format is unusual
            if result.returncode == 0:
                print(f"  ✅ Tesseract installed (version output unavailable)")
            else:
                raise AssertionError("Tesseract command failed")

    except FileNotFoundError:
        raise AssertionError(
            "Tesseract not found.\n"
            "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  Mac: brew install tesseract\n"
            "  Linux: sudo apt install tesseract-ocr"
        )

# ============================================================
# TEST 4: pytesseract (Python bridge to Tesseract)
# ============================================================
def test_pytesseract():
    """
    Check pytesseract Python library AND verify it can communicate
    with the Tesseract engine we installed in Test 3.
    
    We test this by creating a simple white image with text
    and checking if pytesseract can read it.
    """
    import pytesseract
    from PIL import Image, ImageDraw, ImageFont
    # PIL = Python Imaging Library (Pillow)
    # Image: Create and manipulate images
    # ImageDraw: Draw shapes and text on images
    import numpy as np

    # Windows: Tell pytesseract where Tesseract is installed
    # Read the path from our .env file
    tesseract_path = os.getenv("TESSERACT_PATH")
    if tesseract_path and os.path.exists(tesseract_path):
        # Set the path so pytesseract knows where to find tesseract.exe
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"  ℹ️  Using Tesseract at: {tesseract_path}")

    # Create a simple test image: white background, black text
    # Image.new("RGB", (width, height), color): Create new image
    img = Image.new("RGB", (300, 80), color="white")
    
    # Draw text on the image
    draw = ImageDraw.Draw(img)  # Create a drawing canvas on our image
    draw.text(
        (20, 25),           # Position (x, y) to start drawing text
        "Drilling Report",  # The text to draw
        fill="black"        # Text color
    )
    
    # Use pytesseract to extract text from the image
    extracted = pytesseract.image_to_string(img).strip()
    # .strip() removes leading/trailing whitespace and newlines
    
    # Check that it extracted something reasonable
    assert "Drilling" in extracted or "drilling" in extracted or len(extracted) > 3, \
        f"OCR test failed. Extracted: '{extracted}'"
    
    print(f"  ✅ pytesseract working — OCR extracted: '{extracted}'")


# ============================================================
# TEST 5: pdf2image (Convert PDF pages to images for OCR)
# ============================================================
def test_pdf2image():
    """
    Check pdf2image is installed.
    This library converts each page of a scanned PDF into
    an image file, which Tesseract can then read.
    
    Requires Poppler to be installed on the system.
    If this fails:
    - Windows: Download from https://github.com/oschwartz10612/poppler-windows
    - Mac: brew install poppler
    - Linux: sudo apt install poppler-utils
    """
    from pdf2image import convert_from_path
    # We just test the import — we don't need a real PDF for this check
    print("  ✅ pdf2image imported successfully")
    print("     (Poppler must also be installed — see instructions above)")


# ============================================================
# TEST 6: Pillow (Image processing)
# ============================================================
def test_pillow():
    """
    Check Pillow (PIL) is installed.
    Pillow handles image loading, saving, and basic processing.
    It is required by both pytesseract and pdf2image.
    """
    from PIL import Image
    import PIL
    print(f"  ✅ Pillow {PIL.__version__}")


# ============================================================
# TEST 7: LangChain
# ============================================================
def test_langchain():
    """
    Check LangChain core components work.
    LangChain connects our PDF extraction → embeddings → LLM pipeline.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Create a splitter and test it on sample text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,   # Max 100 characters per chunk
        chunk_overlap=10  # 10 character overlap between chunks
    )
    
    # Test with a realistic drilling report sentence
    test_text = (
        "The drilling crew encountered stuck pipe at 9,450 feet. "
        "Mud weight was 10.2 ppg. ROP averaged 45 ft/hr during the morning shift."
    )
    chunks = splitter.split_text(test_text)
    
    assert len(chunks) > 0, "Text splitter returned no chunks"
    print(f"  ✅ LangChain working — split into {len(chunks)} chunk(s)")


# ============================================================
# TEST 8: Sentence Transformers (Embedding model)
# ============================================================
def test_sentence_transformers():
    """
    Check sentence-transformers can create embeddings.
    This converts text into vectors (lists of numbers representing meaning).
    
    NOTE: First run downloads the model file (~90MB) — this is normal.
    The model is cached locally after the first download.
    """
    from sentence_transformers import SentenceTransformer
    
    print("  ⏳ Loading embedding model (first run downloads ~90MB)...")
    
    # Load the embedding model
    # 'all-MiniLM-L6-v2' is small, fast, and runs well on CPU
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create an embedding for a test sentence
    test_sentence = "stuck pipe incident at 9450 feet depth"
    embedding = model.encode(test_sentence)
    
    # This model always produces vectors of exactly 384 numbers
    assert len(embedding) == 384, \
        f"Expected 384 dimensions, got {len(embedding)}"
    
    print(f"  ✅ Embeddings working — '{test_sentence[:30]}...'")
    print(f"     → Vector of {len(embedding)} numbers created")


# ============================================================
# TEST 9: ChromaDB (Persistent vector database)
# ============================================================
# ============================================================
# TEST 9: ChromaDB (Persistent vector database) — Windows Fixed
# ============================================================
def test_chromadb():
    """
    Check ChromaDB is installed and can persist data to disk.
    Fixed for Windows: handles file locking during cleanup gracefully.
    """
    import chromadb
    import shutil
    import time      # We need time to add a small delay on Windows
    import gc        # gc = garbage collector — forces Python to release objects

    test_db_path = "./test_chroma_temp"
    client = None    # Initialize to None so finally block can check it safely

    try:
        # Create ChromaDB client with persistent storage
        client = chromadb.PersistentClient(path=test_db_path)

        # Create a test collection
        collection = client.get_or_create_collection(
            name="test_drilling_reports"
        )

        # Add test documents
        collection.add(
            documents=[
                "Stuck pipe incident occurred at 9450 feet depth",
                "Mud weight was increased to 10.8 ppg at 8000 feet",
                "ROP averaged 45 feet per hour during the morning shift",
                "Lost circulation event at the limestone formation boundary",
            ],
            ids=["doc1", "doc2", "doc3", "doc4"]
        )

        # Search for similar documents
        results = collection.query(
            query_texts=["was there a stuck pipe problem?"],
            n_results=2
        )

        # Verify results
        assert len(results['documents'][0]) == 2, \
            "ChromaDB did not return 2 results"

        print(f"  ✅ ChromaDB working — stored 4 docs, query returned:")
        for i, doc in enumerate(results['documents'][0]):
            print(f"     Result {i+1}: '{doc[:55]}...'")
        print(f"     ✅ Data persisted to disk at: {test_db_path}")

    finally:
        # -------------------------------------------------------
        # Windows-safe cleanup
        # Windows locks files that are in use, so we must:
        # 1. Delete the collection explicitly
        # 2. Release the client object from memory
        # 3. Force garbage collection
        # 4. Wait briefly for Windows to release file handles
        # 5. Then delete the folder
        # -------------------------------------------------------
        try:
            if client is not None:
                # Step 1: Delete the test collection from ChromaDB
                try:
                    client.delete_collection("test_drilling_reports")
                except Exception:
                    pass  # If deletion fails, continue anyway

                # Step 2: Release the client from Python's memory
                del client
                del collection  # Release collection reference too

            # Step 3: Force Python's garbage collector to run
            # This releases any remaining file handles
            gc.collect()

            # Step 4: Wait for Windows to release the file locks
            # 1.5 seconds is usually enough
            time.sleep(1.5)

            # Step 5: Now try to delete the folder
            if os.path.exists(test_db_path):
                shutil.rmtree(test_db_path, ignore_errors=True)
                # ignore_errors=True: Don't crash if some files are still locked

            # Verify cleanup succeeded
            if not os.path.exists(test_db_path):
                print(f"  ✅ Test database cleaned up successfully")
            else:
                # Folder still exists but that is OK — it is just a temp folder
                print(f"  ℹ️  Temp folder could not be fully deleted (Windows lock)")
                print(f"     This is harmless — ChromaDB itself is working correctly")
                print(f"     You can manually delete: {os.path.abspath(test_db_path)}")

        except Exception as cleanup_error:
            # If cleanup fails for any reason, just inform — don't fail the test
            print(f"  ℹ️  Cleanup note: {cleanup_error}")
            print(f"     ChromaDB is working correctly — cleanup issue is harmless")

# ============================================================
# TEST 10: SQLite (Conversation history persistence)
# ============================================================
def test_sqlite():
    """
    Check SQLite is available and working.
    
    SQLite is built into Python — no installation needed.
    We use it to save conversation history (questions and answers)
    to a local database file that persists between sessions.
    
    This test:
    1. Creates a test database
    2. Creates a conversation history table
    3. Inserts a test Q&A entry
    4. Reads it back
    5. Verifies the data is correct
    6. Cleans up
    """
    import sqlite3  # Built into Python — always available
    import datetime  # For timestamps
    
    test_db_path = "./test_sqlite_temp.db"
    
    try:
        # Connect to database (creates the file if it doesn't exist)
        conn = sqlite3.connect(test_db_path)
        
        # cursor: An object that lets us execute SQL commands
        cursor = conn.cursor()
        
        # Create the conversation history table
        # SQL: Structured Query Language — how we talk to databases
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                document_name TEXT,
                question TEXT,
                answer TEXT,
                timestamp TEXT
            )
        """)
        # INTEGER PRIMARY KEY AUTOINCREMENT: Auto-assign unique ID numbers
        # TEXT: Stores text data
        
        # Insert a test conversation entry
        cursor.execute("""
            INSERT INTO conversation_history 
            (session_id, document_name, question, answer, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "session_001",                    # session_id
            "well_A7_day4_report.pdf",        # document_name
            "Was there a stuck pipe?",        # question
            "Yes, stuck pipe at 9450 feet.",  # answer
            datetime.datetime.now().isoformat()  # timestamp
        ))
        # The ? placeholders prevent SQL injection attacks
        # Values are passed separately as a tuple
        
        # Save the changes to disk
        conn.commit()
        
        # Read the data back to verify it was saved
        cursor.execute("SELECT * FROM conversation_history")
        rows = cursor.fetchall()  # Get all rows as a list
        
        assert len(rows) == 1, "Expected 1 row in database"
        assert rows[0][3] == "Was there a stuck pipe?", \
            "Question was not saved correctly"
        
        print(f"  ✅ SQLite working — saved and retrieved conversation:")
        print(f"     Q: '{rows[0][3]}'")
        print(f"     A: '{rows[0][4]}'")
        print(f"     Timestamp: {rows[0][5][:19]}")
        
    finally:
        # Close connection and clean up
        conn.close()
        if os.path.exists(test_db_path):
            os.remove(test_db_path)  # Delete the test database file
            print(f"  ✅ Test database cleaned up")


# ============================================================
# TEST 11: Streamlit
# ============================================================
def test_streamlit():
    """
    Check Streamlit is installed.
    Streamlit is our web app framework — turns Python into a website.
    """
    import streamlit
    print(f"  ✅ Streamlit {streamlit.__version__}")


# ============================================================
# TEST 12: HuggingFace API Token
# ============================================================
def test_env_and_token():
    """
    Check that the .env file exists and contains the HuggingFace token.
    Also verify ChromaDB and SQLite paths are configured.
    """
    # Token check
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token and token.startswith("hf_") and len(token) > 10:
        # Only show last 4 characters — never print full tokens
        print(f"  ✅ HuggingFace token found: hf_...{token[-4:]}")
    else:
        raise AssertionError(
            "HuggingFace token not found or invalid.\n"
            "  Add to .env file: HUGGINGFACEHUB_API_TOKEN=hf_your_token"
        )
    
    # ChromaDB path check
    chroma_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    print(f"  ✅ ChromaDB path configured: {chroma_path}")
    
    # SQLite path check
    sqlite_path = os.getenv("SQLITE_DB_PATH", "./data/conversation_history.db")
    print(f"  ✅ SQLite path configured: {sqlite_path}")
    
    # Windows Tesseract path check
    tesseract_path = os.getenv("TESSERACT_PATH")
    if tesseract_path:
        if os.path.exists(tesseract_path):
            print(f"  ✅ Tesseract path verified: {tesseract_path}")
        else:
            print(f"  ⚠️  Tesseract path in .env not found: {tesseract_path}")
            print(f"      Update TESSERACT_PATH in your .env file")
    else:
        print(f"  ℹ️  No TESSERACT_PATH in .env (OK for Mac/Linux)")


# ============================================================
# MAIN — Run all tests
# ============================================================
if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("  DRILLING REPORT Q&A BOT — SETUP VERIFICATION")
    print("  Full Version: OCR + ChromaDB + SQLite Memory")
    print("="*60 + "\n")
    
    # All tests with their display names
    tests = [
        ("Python Version",           test_python_version),
        ("pdfplumber (Digital PDF)", test_pdfplumber),
        ("Tesseract Engine (OCR)",   test_tesseract_engine),
        ("pytesseract (OCR Bridge)", test_pytesseract),
        ("pdf2image (PDF→Image)",    test_pdf2image),
        ("Pillow (Image Library)",   test_pillow),
        ("LangChain",                test_langchain),
        ("Sentence Transformers",    test_sentence_transformers),
        ("ChromaDB (Vector DB)",     test_chromadb),
        ("SQLite (Chat Memory)",     test_sqlite),
        ("Streamlit (Web App)",      test_streamlit),
        ("API Token & Config",       test_env_and_token),
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for name, test_func in tests:
        print(f"Testing {name}...")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
            failed_tests.append(name)
        print()  # Empty line between tests for readability
    
    # Final summary
    print("="*60)
    print(f"  RESULTS: {passed} passed  |  {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("""
  🎉 ALL 12 TESTS PASSED!
  
  Your environment is fully configured with:
  ✅ Digital PDF extraction (pdfplumber)
  ✅ Scanned PDF support (Tesseract OCR)
  ✅ Persistent vector storage (ChromaDB)
  ✅ Persistent conversation memory (SQLite)
  ✅ AI pipeline (LangChain + HuggingFace)
  ✅ Web interface (Streamlit)
  
  You are ready for Phase 3: Data Collection!
        """)
    else:
        print(f"\n  Fix these failed tests before continuing:")
        for t in failed_tests:
            print(f"  ❌ {t}")
        print("\n  Share the error messages with your mentor for help.\n")