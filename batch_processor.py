# ============================================================
# batch_processor.py
# Professional batch processing pipeline for large PDFs
#
# Processes ANY size PDF completely by reading it in batches
# Memory-safe: never loads the whole PDF into RAM at once
# Progress tracking: shows exactly where processing is
#
# Run with:
# docker exec -it drilling-report-qa python batch_processor.py
# ============================================================

# Suppress ChromaDB telemetry warnings — add these FIRST
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import pdfplumber        # Extract text from digital PDFs
import pytesseract       # OCR for scanned pages
from PIL import Image    # Image handling for OCR
import chromadb          # Vector database
from sentence_transformers import SentenceTransformer
import os
import json
import time
import hashlib           # Generate unique IDs for chunks
from datetime import datetime

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

# Tesseract path for Windows (read from environment)
TESSERACT_PATH = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# ChromaDB storage path
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

# Processing configuration
BATCH_SIZE = 25          # Process 25 pages at a time
CHUNK_SIZE = 600         # Characters per text chunk
CHUNK_OVERLAP = 100      # Overlap between chunks
MIN_TEXT_LENGTH = 30     # Skip pages with less than 30 chars
OCR_TEXT_THRESHOLD = 50  # Pages with less than this need OCR


# -------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------

def generate_chunk_id(text, doc_name, chunk_index):
    """
    Generate a unique ID for each chunk.
    We use MD5 hash of the content to ensure uniqueness.
    
    Why unique IDs matter: ChromaDB requires every document
    to have a unique ID. If we add the same chunk twice,
    it would cause errors. Using content hash prevents duplicates.
    """
    content = f"{doc_name}_{chunk_index}_{text[:50]}"
    # hashlib.md5: Creates a hash (fixed-length fingerprint) of any string
    # encode(): Convert string to bytes (required by hashlib)
    # hexdigest(): Return hash as a hex string
    return hashlib.md5(content.encode()).hexdigest()


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split a long text into overlapping chunks.
    
    Why overlapping? Because answers sometimes span chunk
    boundaries. Overlap ensures we don't miss them.
    
    Example with chunk_size=10, overlap=3:
    Text: "ABCDEFGHIJKLMNOP"
    Chunk 1: "ABCDEFGHIJ"      (positions 0-9)
    Chunk 2: "HIJKLMNOP"       (positions 7-15) ← 3 char overlap
    """
    chunks = []
    start = 0

    while start < len(text):
        # Get chunk from current position
        end = start + chunk_size
        chunk = text[start:end]

        # Only add chunk if it has meaningful content
        if len(chunk.strip()) > MIN_TEXT_LENGTH:
            chunks.append(chunk.strip())

        # Move start forward by (chunk_size - overlap)
        # This creates the overlap between consecutive chunks
        start += (chunk_size - overlap)

    return chunks


def extract_page_text(page, page_number):
    """
    Extract text from a single PDF page.
    
    Strategy:
    1. Try direct text extraction (fast, accurate)
    2. If text is too short, assume scanned → use OCR
    
    This handles mixed PDFs where some pages are digital
    and some are scanned images.
    """
    # Method 1: Direct text extraction
    direct_text = page.extract_text() or ""

    if len(direct_text.strip()) >= OCR_TEXT_THRESHOLD:
        # Enough text found directly — use it
        return direct_text.strip(), "direct"

    # Method 2: OCR fallback for image-heavy pages
    try:
        # Convert the PDF page to a high-resolution image
        # resolution=300: 300 DPI gives good OCR accuracy
        page_image = page.to_image(resolution=300)

        # Convert to PIL Image format that pytesseract expects
        pil_image = page_image.original

        # Run Tesseract OCR on the image
        ocr_text = pytesseract.image_to_string(
            pil_image,
            # lang: Try English first, fall back to Norwegian
            # nor = Norwegian language pack
            lang='eng',
            config='--psm 1'
            # psm 1: Automatic page segmentation with OSD
            # Best for full document pages
        )

        if len(ocr_text.strip()) > MIN_TEXT_LENGTH:
            return ocr_text.strip(), "ocr"

    except Exception as e:
        # If OCR fails for this page, log and continue
        print(f"    ⚠️  OCR failed for page {page_number}: {str(e)[:50]}")

    # Return whatever we have, even if short
    return direct_text.strip(), "direct"


def load_progress(progress_file):
    """
    Load processing progress from a JSON file.
    
    This is crucial for large documents:
    If processing is interrupted (power cut, crash),
    we can resume from where we left off instead of
    starting over from page 1.
    """
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    # No progress file means we start fresh
    return {"last_page_processed": 0, "total_chunks_added": 0}


def save_progress(progress_file, last_page, total_chunks):
    """
    Save current progress to disk after each batch.
    If processing is interrupted, we can resume here.
    """
    progress = {
        "last_page_processed": last_page,
        "total_chunks_added": total_chunks,
        "last_updated": datetime.now().isoformat()
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


# -------------------------------------------------------
# MAIN PROCESSING CLASS
# -------------------------------------------------------

class DocumentBatchProcessor:
    """
    Processes large PDF documents in batches.
    
    This class handles the complete pipeline:
    PDF → text extraction → chunking → embedding → ChromaDB
    
    It is memory-safe (processes one batch at a time)
    and resumable (tracks progress to file)
    """

    def __init__(self):
        """
        Initialize the processor with ChromaDB and embedding model.
        __init__ runs automatically when we create a processor object.
        """
        print("Initializing Document Batch Processor...")

        # Initialize ChromaDB client
        # PersistentClient: Saves everything to disk automatically
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        print(f"  ChromaDB connected at: {CHROMA_PATH}")

        # Load the embedding model
        # This model converts text → vectors (384 numbers)
        print("   Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   Embedding model loaded")

        print("   Processor ready\n")

    def get_or_create_collection(self, collection_name):
        """
        Get existing ChromaDB collection or create a new one.
        A collection is like a table — it holds all chunks
        from one document.
        """
        # Sanitize collection name
        # ChromaDB collection names can't have spaces or special chars
        safe_name = collection_name.replace(" ", "_") \
                                   .replace(".", "_") \
                                   .replace("-", "_") \
                                   .lower()

        collection = self.chroma_client.get_or_create_collection(
            name=safe_name,
            # metadata: Extra info stored with the collection
            metadata={
                "document_name": collection_name,
                "created_at": datetime.now().isoformat()
            }
        )
        return collection

    def process_batch(self, pages, batch_num, doc_name, collection):
        """
        Process one batch of pages:
        1. Extract text from each page
        2. Split into chunks
        3. Create embeddings
        4. Store in ChromaDB

        Returns: number of chunks added in this batch
        """
        batch_text_parts = []    # Collect text from all pages in batch
        extraction_stats = {"direct": 0, "ocr": 0, "empty": 0}

        # Step 1: Extract text from each page in this batch
        for page_num, page in pages:
            text, method = extract_page_text(page, page_num)

            if len(text.strip()) > MIN_TEXT_LENGTH:
                # Add page number context to help with answers
                # This way the LLM can say "from page 45..."
                contextualized_text = f"[Page {page_num}]\n{text}"
                batch_text_parts.append(contextualized_text)
                extraction_stats[method] += 1
            else:
                extraction_stats["empty"] += 1

        if not batch_text_parts:
            return 0  # Nothing to process in this batch

        # Step 2: Combine all page texts and split into chunks
        full_batch_text = "\n\n".join(batch_text_parts)
        chunks = split_text_into_chunks(full_batch_text)

        if not chunks:
            return 0

        # Step 3: Create embeddings for all chunks in this batch
        # encode(): Convert list of text strings → list of vectors
        embeddings = self.embedding_model.encode(
            chunks,
            # show_progress_bar: Show progress for large batches
            show_progress_bar=False,
            # batch_size: Process 32 chunks at a time internally
            batch_size=32
        )

        # Convert embeddings to list format (ChromaDB requires lists)
        embeddings_list = embeddings.tolist()

        # Step 4: Generate unique IDs for each chunk
        ids = [
            generate_chunk_id(chunk, doc_name, f"batch{batch_num}_chunk{i}")
            for i, chunk in enumerate(chunks)
        ]

        # Step 5: Create metadata for each chunk
        # Metadata helps us filter and understand search results
        metadatas = [
            {
                "document": doc_name,
                "batch": batch_num,
                "chunk_index": i,
                "char_count": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]

        # Step 6: Add everything to ChromaDB
        # We add in smaller sub-batches of 100 to avoid timeouts
        sub_batch_size = 100
        for i in range(0, len(chunks), sub_batch_size):
            collection.add(
                documents=chunks[i:i+sub_batch_size],
                embeddings=embeddings_list[i:i+sub_batch_size],
                ids=ids[i:i+sub_batch_size],
                metadatas=metadatas[i:i+sub_batch_size]
            )

        print(f"    Pages processed: direct={extraction_stats['direct']}, "
              f"ocr={extraction_stats['ocr']}, "
              f"empty={extraction_stats['empty']}")
        print(f"    Chunks created: {len(chunks)}")

        return len(chunks)

    def process_pdf(self, pdf_path):
        """
        Main method: Process an entire PDF file in batches.
        Handles progress tracking and resumption.
        """
        doc_name = os.path.basename(pdf_path)
        progress_file = f"data/processed/.progress_{doc_name}.json"
        os.makedirs("data/processed", exist_ok=True)

        print(f"\n{'='*60}")
        print(f"PROCESSING: {doc_name}")
        print(f"{'='*60}")

        # Check file exists
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}")
            return

        # Get file size
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")

        # Load progress (in case we are resuming)
        progress = load_progress(progress_file)
        start_page = progress["last_page_processed"]
        total_chunks = progress["total_chunks_added"]

        if start_page > 0:
            print(f"⏩ Resuming from page {start_page + 1}")
            print(f"   (Already indexed {total_chunks} chunks)")

        # Get or create ChromaDB collection for this document
        collection_name = doc_name.replace(".pdf", "")
        collection = self.get_or_create_collection(collection_name)

        # Open the PDF and process in batches
        start_time = time.time()

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
            print(f"Batch size: {BATCH_SIZE} pages per batch")
            estimated_batches = (total_pages - start_page) // BATCH_SIZE + 1
            print(f"Estimated batches: {estimated_batches}")
            print(f"Starting processing...\n")

            # Process pages in batches
            batch_num = 0

            for batch_start in range(start_page, total_pages, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_pages)
                batch_num += 1

                print(f"Batch {batch_num}/{estimated_batches}: "
                      f"Pages {batch_start+1} to {batch_end}")

                # Get pages for this batch
                # We pair each page with its page number (1-indexed)
                batch_pages = [
                    (i + 1, pdf.pages[i])
                    for i in range(batch_start, batch_end)
                ]

                # Process this batch
                chunks_added = self.process_batch(
                    batch_pages,
                    batch_num,
                    doc_name,
                    collection
                )
                total_chunks += chunks_added

                # Save progress after every batch
                # If processing is interrupted, we resume from here
                save_progress(progress_file, batch_end, total_chunks)

                # Calculate and show progress
                pages_done = batch_end - start_page
                pages_total = total_pages - start_page
                percent = (pages_done / pages_total) * 100
                elapsed = time.time() - start_time
                pages_per_second = pages_done / elapsed if elapsed > 0 else 0
                remaining_pages = pages_total - pages_done
                eta_seconds = remaining_pages / pages_per_second \
                    if pages_per_second > 0 else 0
                eta_minutes = eta_seconds / 60

                print(f"    Progress: {percent:.1f}% | "
                      f"Total chunks: {total_chunks} | "
                      f"ETA: {eta_minutes:.1f} min\n")

        # Processing complete
        total_time = (time.time() - start_time) / 60

        print(f"{'='*60}")
        print(f"✅ COMPLETE: {doc_name}")
        print(f"   Total pages processed : {total_pages}")
        print(f"   Total chunks indexed  : {total_chunks}")
        print(f"   Processing time       : {total_time:.1f} minutes")
        print(f"   ChromaDB collection   : {collection_name}")
        print(f"{'='*60}\n")

        # Clean up progress file after successful completion
        if os.path.exists(progress_file):
            os.remove(progress_file)

        return {
            "document": doc_name,
            "pages": total_pages,
            "chunks": total_chunks,
            "time_minutes": total_time
        }

    def process_text_file(self, txt_path):
        """
        Process a plain text file (our simulated DDR reports).
        Text files are much simpler — just read and chunk.
        """
        doc_name = os.path.basename(txt_path)

        print(f"\nProcessing text file: {doc_name}")

        # Read the text file
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()

        if len(text.strip()) < MIN_TEXT_LENGTH:
            print(f"  ⚠️  File appears empty: {txt_path}")
            return

        # Split into chunks
        chunks = split_text_into_chunks(text)

        if not chunks:
            print(f"  ⚠️  No chunks created from: {doc_name}")
            return

        # Create embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        embeddings_list = embeddings.tolist()

        # Generate unique IDs
        ids = [
            generate_chunk_id(chunk, doc_name, i)
            for i, chunk in enumerate(chunks)
        ]

        # Create metadata
        metadatas = [
            {"document": doc_name, "chunk_index": i, "char_count": len(chunk)}
            for i, chunk in enumerate(chunks)
        ]

        # Get collection and add documents
        collection_name = doc_name.replace(".txt", "").replace(".", "_")
        collection = self.get_or_create_collection(collection_name)
        collection.add(
            documents=chunks,
            embeddings=embeddings_list,
            ids=ids,
            metadatas=metadatas
        )

        print(f"  ✅ {doc_name}: {len(chunks)} chunks indexed")
        return len(chunks)


# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  FULL DOCUMENT BATCH PROCESSOR")
    print("  Processes ALL pages — no skipping")
    print("="*60)

    # Initialize the processor
    processor = DocumentBatchProcessor()

    # Define all files to process
    # Text files: our 10 simulated DDR reports
    # PDF files: our 2 real Equinor documents
    data_dir = "data/raw_reports"

    txt_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.txt')
    ])

    pdf_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('.pdf')
    ])

    print(f"\nFiles to process:")
    print(f"  Text files : {len(txt_files)}")
    print(f"  PDF files  : {len(pdf_files)}")
    print(f"  Total      : {len(txt_files) + len(pdf_files)} documents\n")

    # Track overall results
    results = []

    # Step 1: Process all text files (fast)
    print("STEP 1: Processing simulated DDR text files...")
    print("-" * 40)
    for txt_path in txt_files:
        chunks = processor.process_text_file(txt_path)
        if chunks:
            results.append({
                "file": os.path.basename(txt_path),
                "chunks": chunks
            })

    print(f"\n✅ All text files processed\n")

    # Step 2: Process PDF files (slower — batch processing)
    print("STEP 2: Processing real Equinor PDF documents...")
    print("This will take several minutes for the large PDF.")
    print("Progress is saved after every batch — safe to interrupt.\n")
    print("-" * 40)

    for pdf_path in pdf_files:
        result = processor.process_pdf(pdf_path)
        if result:
            results.append(result)

    # Final summary
    print("\n" + "="*60)
    print("  ALL PROCESSING COMPLETE")
    print("="*60)
    total_chunks = sum(r.get('chunks', 0) for r in results)
    print(f"  Documents processed : {len(results)}")
    print(f"  Total chunks indexed: {total_chunks:,}")
    print(f"\n  Your ChromaDB now contains ALL content from:")
    for r in results:
        name = r.get('file') or r.get('document', 'unknown')
        chunks = r.get('chunks', 0)
        print(f"    ✅ {name}: {chunks} chunks")
    print(f"\n  Ready for Phase 4: Data Preprocessing!")
    print("="*60 + "\n")