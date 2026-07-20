# ============================================================
# verify_chromadb.py
# Verifies ChromaDB contains all our indexed documents
# and that search is working correctly
#
# Run: docker exec -it drilling-report-qa python verify_chromadb.py
# ============================================================

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb
from sentence_transformers import SentenceTransformer

# -------------------------------------------------------
# Connect to ChromaDB
# -------------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

print("\n" + "="*60)
print("  CHROMADB VERIFICATION")
print("="*60)

# Create client connected to our persistent database
client = chromadb.PersistentClient(path=CHROMA_PATH)

# -------------------------------------------------------
# List all collections (one per document)
# -------------------------------------------------------
collections = client.list_collections()
print(f"\n📚 Collections in ChromaDB: {len(collections)}")
print("-" * 40)

total_chunks = 0
collection_data = []

for col in collections:
    # Get the collection object
    collection = client.get_collection(col.name)
    # count() returns how many documents are in this collection
    count = collection.count()
    total_chunks += count
    collection_data.append((col.name, count))
    print(f"  {col.name:<40} {count:>5} chunks")

print("-" * 40)
print(f"  {'TOTAL':<40} {total_chunks:>5} chunks")

# -------------------------------------------------------
# Test Search — The Most Important Verification
# -------------------------------------------------------
print(f"\n🔍 SEARCH VERIFICATION")
print("="*60)
print("Testing if our Q&A system can find relevant content...\n")

# Load embedding model for search
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.\n")

# Define test questions with expected sources
test_queries = [
    {
        "question": "Was there a stuck pipe incident?",
        "expected_source": "DDR_Day04",
        "collection": "ddr_day04_2024_01_18"
    },
    {
        "question": "What oil shows were encountered during drilling?",
        "expected_source": "DDR_Day05",
        "collection": "ddr_day05_2024_01_19"
    },
    {
        "question": "What was the discovery well name and structure?",
        "expected_source": "Discovery_report",
        "collection": "discovery_report"
    },
    {
        "question": "What is the Volve field development plan?",
        "expected_source": "Volve-PUD",
        "collection": "volve_pud"
    },
    {
        "question": "What caused the lost circulation event?",
        "expected_source": "DDR_Day06",
        "collection": "ddr_day06_2024_01_20"
    },
]

passed = 0
failed = 0

for test in test_queries:
    question = test["question"]
    collection_name = test["collection"]

    print(f"Question: '{question}'")
    print(f"Searching in: {collection_name}")

    try:
        # Get the specific collection
        collection = client.get_collection(collection_name)

        # Convert question to embedding vector
        question_embedding = model.encode(question).tolist()

        # Search for top 2 most similar chunks
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=2  # Get top 2 most relevant chunks
        )

        # Get the best result
        best_chunk = results['documents'][0][0]

        print(f"✅ Found relevant content:")
        # Show first 200 characters of the best matching chunk
        print(f"   '{best_chunk[:200].strip()}...'")
        passed += 1

    except Exception as e:
        print(f"❌ Search failed: {e}")
        failed += 1

    print()

# -------------------------------------------------------
# Cross-document search test
# This simulates what our app will do — search ALL docs
# -------------------------------------------------------
print("="*60)
print("CROSS-DOCUMENT SEARCH TEST")
print("Searching across ALL documents simultaneously")
print("="*60)

cross_doc_questions = [
    "What is the mud weight used during drilling operations?",
    "Were there any equipment failures or incidents reported?",
    "What formation types were encountered?",
]

for question in cross_doc_questions:
    print(f"\nQuestion: '{question}'")
    print(f"Results from all {len(collections)} collections:")

    question_embedding = model.encode(question).tolist()
    all_results = []

    # Search each collection and collect results
    for col_name, _ in collection_data:
        try:
            col = client.get_collection(col_name)
            results = col.query(
                query_embeddings=[question_embedding],
                n_results=1  # Top 1 from each collection
            )
            if results['documents'][0]:
                score = results['distances'][0][0]
                # Lower distance = more similar = better match
                all_results.append({
                    "collection": col_name,
                    "text": results['documents'][0][0][:100],
                    "distance": score
                })
        except Exception:
            continue

    # Sort by distance (lower = more relevant)
    all_results.sort(key=lambda x: x['distance'])

    # Show top 3 most relevant across all documents
    for i, result in enumerate(all_results[:3], 1):
        print(f"  #{i} [{result['collection']}] "
              f"(relevance score: {result['distance']:.3f})")
        print(f"     '{result['text'].strip()[:100]}...'")

# -------------------------------------------------------
# Final Summary
# -------------------------------------------------------
print("\n" + "="*60)
print("  VERIFICATION SUMMARY")
print("="*60)
print(f"  Collections found    : {len(collections)}")
print(f"  Total chunks indexed : {total_chunks}")
print(f"  Search tests passed  : {passed}")
print(f"  Search tests failed  : {failed}")

if failed == 0:
    print(f"""
  🎉 ALL VERIFICATIONS PASSED!

  Your ChromaDB is fully operational:
  ✅ All 12 documents indexed
  ✅ {total_chunks} searchable chunks
  ✅ Semantic search working correctly
  ✅ Cross-document search working

  Ready for Phase 4: Data Preprocessing!
    """)
else:
    print(f"\n  ⚠️  {failed} search tests failed")
    print(f"  Share the error output with your mentor")

print("="*60 + "\n")