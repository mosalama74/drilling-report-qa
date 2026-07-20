# ============================================================
#
# Handles text extraction from ALL document types:
# - Digital PDFs (pdfplumber — fast, accurate)
# - Scanned PDFs (pdf2image + pytesseract OCR)
# - Plain text files (direct read)
#
# Auto-detects document type and uses correct method.
# ============================================================

import os
import pdfplumber          # Digital PDF text extraction
import pytesseract         # Python bridge to Tesseract OCR
from pdf2image import convert_from_path  # PDF pages → images for OCR
from PIL import Image      # Image handling
from src.text_cleaner import TextCleaner  # Our cleaning pipeline

# Configure Tesseract path from environment variable
# Inside Docker: /usr/bin/tesseract
# On Windows: C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Minimum characters to consider a page "readable"
# Pages with less than this are treated as scanned/image pages
MIN_CHARS_FOR_DIGITAL = 50


class DocumentExtractor:
    """
    Extracts clean text from any document type.

    Automatically detects:
    - Is this a PDF or text file?
    - Is the PDF digital or scanned?
    - Which pages need OCR?

    Then applies our TextCleaner to produce clean output.
    """

    def __init__(self):
        """Initialize extractor with text cleaner"""
        self.cleaner = TextCleaner()

    def detect_document_type(self, file_path):
        """
        Detect what type of document we are dealing with.

        Returns: 'text', 'digital_pdf', or 'scanned_pdf'
        """
        # Get file extension (lowercase for consistency)
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        if extension == '.txt':
            return 'text'

        if extension == '.pdf':
            # Check if PDF has extractable text
            return self._check_pdf_type(file_path)

        # Unknown type
        return 'unknown'

    def _check_pdf_type(self, pdf_path):
        """
        Check if a PDF is digital or scanned.

        Strategy: Read the first 3 pages and try to extract text.
        If we get meaningful text → digital PDF.
        If text is empty or very short → scanned PDF.
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Check up to first 3 pages
                pages_to_check = min(3, len(pdf.pages))
                total_text = ""

                for i in range(pages_to_check):
                    page_text = pdf.pages[i].extract_text() or ""
                    total_text += page_text

                # If we got meaningful text, it is digital
                if len(total_text.strip()) > MIN_CHARS_FOR_DIGITAL:
                    return 'digital_pdf'
                else:
                    return 'scanned_pdf'

        except Exception:
            # If we cannot even open it, try OCR
            return 'scanned_pdf'

    def extract_from_text_file(self, file_path):
        """
        Extract text from a plain .txt file.

        This is the simplest case — just read the file.
        Then clean the text.
        """
        doc_name = os.path.basename(file_path)

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # errors='replace': Replace unreadable characters with ?
            # instead of crashing on encoding errors
            raw_text = f.read()

        # Clean the text
        clean_text = self.cleaner.clean(raw_text, doc_name)

        return {
            "text": clean_text,
            "pages": 1,         # Text files treated as 1 "page"
            "method": "direct_read",
            "document_name": doc_name
        }

    def extract_from_digital_pdf(self, file_path):
        """
        Extract text from a digital PDF using pdfplumber.

        Digital PDF = text is encoded in the PDF file itself.
        We can extract it directly — fast and accurate.
        """
        doc_name = os.path.basename(file_path)
        all_pages_text = []
        pages_processed = 0
        pages_ocr_needed = 0

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text from this page
                page_text = page.extract_text() or ""

                if len(page_text.strip()) >= MIN_CHARS_FOR_DIGITAL:
                    # Good digital text — add with page reference
                    # Adding page number helps the LLM cite sources
                    all_pages_text.append(
                        f"[Page {page_num} of {total_pages}]\n{page_text}"
                    )
                    pages_processed += 1

                else:
                    # This page seems to be an image — use OCR
                    ocr_text = self._ocr_single_page(page, page_num)
                    if ocr_text:
                        all_pages_text.append(
                            f"[Page {page_num} of {total_pages} - OCR]\n"
                            f"{ocr_text}"
                        )
                        pages_ocr_needed += 1

        # Join all pages with double newline between them
        full_text = "\n\n".join(all_pages_text)

        # Clean the complete text
        clean_text = self.cleaner.clean(full_text, doc_name)

        print(f"    Extraction complete: {pages_processed} digital pages, "
              f"{pages_ocr_needed} OCR pages")

        return {
            "text": clean_text,
            "pages": total_pages,
            "method": "digital_pdf",
            "pages_ocr": pages_ocr_needed,
            "document_name": doc_name
        }

    def extract_from_scanned_pdf(self, file_path):
        """
        Extract text from a scanned PDF using OCR.

        Scanned PDF = pages are images, no embedded text.
        We must convert each page to an image then run OCR.

        This is slower than digital extraction but handles
        any scanned document correctly.
        """
        doc_name = os.path.basename(file_path)
        all_pages_text = []

        print(f"    Running OCR on scanned PDF: {doc_name}")
        print(f"    This may take several minutes...")

        # Convert PDF pages to images
        # dpi=300: High resolution for better OCR accuracy
        # fmt='jpeg': JPEG format (faster than PNG for OCR)
        images = convert_from_path(
            file_path,
            dpi=300,
            fmt='jpeg'
        )

        total_pages = len(images)
        print(f"    Total pages to OCR: {total_pages}")

        for page_num, image in enumerate(images, start=1):
            # Run Tesseract OCR on this page image
            page_text = self._ocr_image(image, page_num)

            if page_text:
                all_pages_text.append(
                    f"[Page {page_num} of {total_pages} - OCR]\n{page_text}"
                )

            # Show progress every 10 pages
            if page_num % 10 == 0:
                print(f"    OCR progress: {page_num}/{total_pages} pages")

        full_text = "\n\n".join(all_pages_text)
        clean_text = self.cleaner.clean(full_text, doc_name)

        return {
            "text": clean_text,
            "pages": total_pages,
            "method": "scanned_pdf_ocr",
            "document_name": doc_name
        }

    def _ocr_single_page(self, pdfplumber_page, page_num):
        """
        Run OCR on a single page from a digital PDF.

        Used when a digital PDF has some image-only pages
        (like figures, scanned attachments embedded in a PDF).
        """
        try:
            # Convert the pdfplumber page to a high-res image
            page_image = pdfplumber_page.to_image(resolution=300)
            # .original gives us the PIL Image object
            pil_image = page_image.original

            # Run OCR
            text = self._ocr_image(pil_image, page_num)
            return text

        except Exception as e:
            print(f"    ⚠️  OCR failed for page {page_num}: {str(e)[:60]}")
            return ""

    def _ocr_image(self, pil_image, page_num):
        """
        Run Tesseract OCR on a PIL image.

        Configuration:
        lang='eng': Use English language model
        --psm 1: Automatic page segmentation with orientation detection
                 Best for full document pages
        --oem 3: Use both legacy and LSTM OCR engines
                 Most accurate combination
        """
        try:
            text = pytesseract.image_to_string(
                pil_image,
                lang='eng',
                config='--psm 1 --oem 3'
            )
            return text.strip()

        except Exception as e:
            print(f"    ⚠️  OCR error on page {page_num}: {str(e)[:60]}")
            return ""

    def extract(self, file_path):
        """
        MAIN METHOD: Extract text from any document.

        Automatically detects document type and uses
        the correct extraction method.

        This is the ONLY function you need to call from outside.
        Everything else is handled internally.

        Usage:
            extractor = DocumentExtractor()
            result = extractor.extract("path/to/report.pdf")
            text = result["text"]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")

        doc_name = os.path.basename(file_path)
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        print(f"\n  Extracting: {doc_name} ({file_size_mb:.1f} MB)")

        # Auto-detect document type
        doc_type = self.detect_document_type(file_path)
        print(f"  Detected type: {doc_type}")

        # Route to correct extraction method
        if doc_type == 'text':
            return self.extract_from_text_file(file_path)

        elif doc_type == 'digital_pdf':
            return self.extract_from_digital_pdf(file_path)

        elif doc_type == 'scanned_pdf':
            return self.extract_from_scanned_pdf(file_path)

        else:
            raise ValueError(
                f"Unsupported document type: {doc_type}\n"
                f"Supported types: .txt, .pdf"
            )