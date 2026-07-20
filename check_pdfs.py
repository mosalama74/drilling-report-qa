# check_pdfs.py
# Checks our real PDF files are readable
# Run with: docker exec -it drilling-report-qa python check_pdfs.py

import pdfplumber
import os

# Note: Volve PUD has a space before .pdf — must match exactly
pdf_files = [
    'data/raw_reports/Discovery_report.pdf',
    'data/raw_reports/Volve-PUD.pdf',
]

for pdf_path in pdf_files:
    print(f"\nChecking: {pdf_path}")
    print("-" * 50)

    if not os.path.exists(pdf_path):
        print(f"NOT FOUND — check filename exactly")
        continue

    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"Total pages: {num_pages}")

            # Try to extract text from page 1
            page1_text = pdf.pages[0].extract_text() or ''

            if len(page1_text.strip()) > 50:
                # Digital PDF — text is directly extractable
                print(f"Type: DIGITAL PDF (text extractable directly)")
                print(f"Preview (first 300 chars):")
                print(page1_text[:300])
            else:
                # Scanned PDF — needs OCR
                print(f"Type: SCANNED PDF (needs OCR)")
                print(f"Text extracted from page 1: '{page1_text[:100]}'")
                print(f"Our OCR pipeline will handle this automatically")

    except Exception as e:
        print(f"Error reading PDF: {e}")

print("\n" + "="*50)
print("PDF CHECK COMPLETE")
print("="*50)