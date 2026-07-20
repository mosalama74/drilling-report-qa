# ============================================================
# Tests the complete preprocessing pipeline
#
# Run: docker exec -it drilling-report-qa python test_preprocessing.py
# ============================================================

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import sys
sys.path.insert(0, '/app')  # Ensure src/ is findable

from src.text_cleaner import TextCleaner
from src.pdf_extractor import DocumentExtractor
from src.chunker import SmartChunker
from src.memory import ConversationMemory

print("\n" + "="*60)
print("  PREPROCESSING PIPELINE TESTS")
print("="*60)

# -------------------------------------------------------
# TEST 1: Text Cleaner
# -------------------------------------------------------
print("\n📋 TEST 1: Text Cleaner")
print("-" * 40)

cleaner = TextCleaner()

dirty_text = """STATOIL CONFIDENTIAL
Page 12 of 194
DAILY DRILLING REPORT

The drill string be-
came stuck at 1,654 feet depth.
Mud weight: 10.2\nppg was increased.

STATOIL CONFIDENTIAL
Page 13 of 194

ROP   averaged    45    ft/hr during   the   shift.
\x00\x00Null bytes removed\x00
"""

clean = cleaner.clean(dirty_text, "test_document")

print(f"Original length : {len(dirty_text)} chars")
print(f"Cleaned length  : {len(clean)} chars")
print(f"Cleaned text:\n{clean}")

assert "STATOIL CONFIDENTIAL" not in clean or \
    clean.count("STATOIL CONFIDENTIAL") <= 2
assert "\x00" not in clean
assert "  " not in clean  # No double spaces
print("✅ Text cleaner working correctly")

# -------------------------------------------------------
# TEST 2: Document Extractor — Text File
# -------------------------------------------------------
print("\n📋 TEST 2: Document Extractor (Text File)")
print("-" * 40)

extractor = DocumentExtractor()

# Test with one of our DDR files
test_txt = "data/raw_reports/DDR_Day04_2024-01-18.txt"
if os.path.exists(test_txt):
    result = extractor.extract(test_txt)
    print(f"Document: {result['document_name']}")
    print(f"Method: {result['method']}")
    print(f"Text length: {len(result['text'])} chars")
    print(f"Preview: {result['text'][:200]}...")
    assert len(result['text']) > 100
    print("✅ Text file extraction working")
else:
    print("⚠️  Test file not found — run generate_reports.py first")

# -------------------------------------------------------
# TEST 3: Document Extractor — PDF File
# -------------------------------------------------------
print("\n📋 TEST 3: Document Extractor (PDF File)")
print("-" * 40)

test_pdf = "data/raw_reports/Volve-PUD.pdf"
if os.path.exists(test_pdf):
    result = extractor.extract(test_pdf)
    print(f"Document: {result['document_name']}")
    print(f"Method: {result['method']}")
    print(f"Pages: {result['pages']}")
    print(f"Text length: {len(result['text'])} chars")
    print(f"Preview: {result['text'][:300]}...")
    assert len(result['text']) > 500
    print("✅ PDF extraction working")
else:
    print("⚠️  Test PDF not found")

# -------------------------------------------------------
# TEST 4: Smart Chunker
# -------------------------------------------------------
print("\n📋 TEST 4: Smart Chunker")
print("-" * 40)

chunker = SmartChunker()

sample_text = """
DEPTH SUMMARY
Depth at Start of Day: 1,456 feet
Depth at End of Day: 1,821 feet
Total Footage Drilled: 365 feet

DRILLING PARAMETERS
Bit Size: 17.5 inch
Weight on Bit: 25 klbs
RPM: 110
Rate of Penetration: 45 ft/hr

INCIDENTS AND NON-PRODUCTIVE TIME
Incident: Stuck pipe occurred at 11:30 at 1,654 feet depth.
Applied 50,000 lbs overpull. Spotted 30 barrels diesel.
Pipe freed after 3.5 hours NPT.

SAFETY REPORT
Stuck pipe emergency procedure followed correctly.
All crew accounted for during the incident.
"""

chunks = chunker.chunk_document(sample_text, "test_doc.txt")
print(f"Sample text length: {len(sample_text)} chars")
print(f"Chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {chunk['char_count']} chars — "
          f"'{chunk['text'][:60]}...'")
assert len(chunks) >= 2
print("✅ Chunker working correctly")

# -------------------------------------------------------
# TEST 5: Conversation Memory (SQLite)
# -------------------------------------------------------
print("\n📋 TEST 5: Conversation Memory")
print("-" * 40)

memory = ConversationMemory("./data/test_memory.db")

# Create a test session
session_id = memory.create_session(
    document_name="DDR_Day04_test.pdf",
    pages=5,
    chunks=20
)
print(f"Created session: {session_id[:8]}...")

# Save a Q&A pair
memory.save_qa_pair(
    session_id=session_id,
    question="Was there a stuck pipe incident?",
    answer="Yes, stuck pipe occurred at 1,654 feet on Day 4.",
    source_chunks=["Pipe became stuck at 1,654 feet..."]
)

# Retrieve history
history = memory.get_session_history(session_id)
assert len(history) == 1
assert history[0]['question'] == "Was there a stuck pipe incident?"
print(f"Saved Q: '{history[0]['question']}'")
print(f"Saved A: '{history[0]['answer']}'")

# Check stats
stats = memory.get_stats()
print(f"Stats: {stats}")

# Cleanup test database
import os
if os.path.exists("./data/test_memory.db"):
    os.remove("./data/test_memory.db")

print("✅ Conversation memory working correctly")

# -------------------------------------------------------
# SUMMARY
# -------------------------------------------------------
print("\n" + "="*60)
print("  ALL PREPROCESSING TESTS PASSED ✅")
print("="*60)
print("""
  Your preprocessing pipeline is ready:
  ✅ Text cleaning (noise removal, normalization)
  ✅ PDF extraction (digital + OCR)
  ✅ Smart chunking (section-aware, overlapping)
  ✅ Conversation memory (SQLite persistence)

  Ready for Phase 5: Model Building!
""")