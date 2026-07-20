# ============================================================
#
# Handles all text cleaning operations.
# Called after text is extracted from PDF or TXT files.
# Returns clean, normalized text ready for chunking.
# ============================================================

import re           # Regular expressions — for pattern matching in text
import unicodedata  # For handling special characters from different languages


class TextCleaner:
    """
    Cleans raw text extracted from petroleum documents.

    Why a class? Because we want to group all cleaning
    operations together and share configuration between them.
    Think of it as a cleaning toolkit in one box.
    """

    def __init__(self):
        """
        Initialize the cleaner with petroleum-specific settings.
        These are patterns we want to REMOVE from raw text.
        """

        # Patterns that indicate page headers/footers to remove
        # re.compile: Pre-compile patterns for speed (faster than compiling every call)
        # re.IGNORECASE: Match regardless of uppercase/lowercase
        # re.MULTILINE: ^ and $ match start/end of each LINE not just the whole string
        self.header_footer_patterns = [
            # Page number patterns
            re.compile(r'^\s*[Pp]age\s+\d+\s*(of\s+\d+)?\s*$', re.MULTILINE),
            re.compile(r'^\s*-\s*\d+\s*-\s*$', re.MULTILINE),
            re.compile(r'^\s*\d+\s*$', re.MULTILINE),

            # Common petroleum document headers
            re.compile(r'CONFIDENTIAL\s*$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'STATOIL\s+CONFIDENTIAL', re.IGNORECASE),
            re.compile(r'PROPRIETARY\s+INFORMATION', re.IGNORECASE),

            # Document identifiers repeated on every page
            re.compile(r'DAILY\s+DRILLING\s+REPORT\s*$', re.MULTILINE),
            re.compile(r'DDR\s+\d+\s*$', re.MULTILINE),
        ]

        # Petroleum units we must PRESERVE exactly
        # These patterns ensure units stay attached to their numbers
        self.petroleum_units = [
            'ppg',   # pounds per gallon (mud weight)
            'psi',   # pounds per square inch (pressure)
            'bbl',   # barrels
            'bbls',  # barrels (plural)
            'ft/hr', # feet per hour (Rate of Penetration)
            'rpm',   # rotations per minute
            'klbs',  # kilo-pounds (Weight on Bit)
            'SG',    # specific gravity
            'ppb',   # pounds per barrel
            'bph',   # barrels per hour
            'ECD',   # Equivalent Circulating Density
            'ROP',   # Rate of Penetration
            'WOB',   # Weight on Bit
            'NPT',   # Non-Productive Time
        ]

    def remove_null_bytes(self, text):
        """
        Remove null bytes and other control characters.

        Null bytes (\x00) are invisible characters that sometimes
        appear in PDF-extracted text. They break string operations
        and confuse the LLM.

        Control characters are non-printable characters like
        \x01, \x02 etc. We keep \n (newline) and \t (tab).
        """
        # Remove null bytes specifically
        text = text.replace('\x00', '')

        # Remove other control characters EXCEPT newline(\n) and tab(\t)
        # \x00-\x08: Control chars before tab
        # \x0b-\x0c: Vertical tab and form feed
        # \x0e-\x1f: Other control chars
        # [^\x09\x0a\x0d\x20-\x7e\x80-\xff]: Keep printable + Unicode
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        return text

    def fix_hyphenated_words(self, text):
        """
        Fix words broken across lines with hyphens.

        PDFs with column layouts often break long words:
        "forma-\ntion" should be "formation"
        "differ-\nential" should be "differential"

        But we must NOT fix intentional hyphens like:
        "oil-based mud" or "PDC-bit"
        """
        # Pattern: word ending in hyphen, followed by newline, 
        # followed by lowercase letter (indicates broken word)
        # The lowercase after newline is key — broken words continue lowercase
        text = re.sub(r'(\w)-\n([a-z])', r'\1\2', text)

        return text

    def remove_headers_footers(self, text):
        """
        Remove repeated page headers and footers.

        In a 194-page document, "STATOIL CONFIDENTIAL" appears
        194 times. Each occurrence adds noise without meaning.
        After removing, our chunks contain only real content.
        """
        for pattern in self.header_footer_patterns:
            text = pattern.sub('', text)

        return text

    def normalize_whitespace(self, text):
        """
        Normalize whitespace while preserving paragraph structure.

        Rules:
        - Multiple spaces → single space
        - More than 2 consecutive newlines → exactly 2 newlines
        - Tabs → single space
        - Keep single and double newlines (paragraph breaks)

        Why preserve paragraph breaks? Because they help the
        chunker split text at natural boundaries.
        """
        # Replace tabs with single space
        text = text.replace('\t', ' ')

        # Replace multiple spaces with single space
        # \s here means space/tab only (not newline) — we handle newlines separately
        text = re.sub(r' {2,}', ' ', text)

        # Replace 3+ consecutive newlines with exactly 2
        # This preserves paragraph breaks without excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove spaces at the beginning and end of each line
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)

        return text

    def normalize_unicode(self, text):
        """
        Normalize Unicode characters.

        PDFs sometimes use special Unicode characters for things
        like quotation marks, dashes, and apostrophes.
        We normalize these to standard ASCII equivalents.

        Examples:
        " " (Unicode quotes) → " " (standard quotes)
        – — (Unicode dashes) → - (standard hyphen)
        © ® → removed (copyright symbols, not content)
        """
        # Normalize Unicode to NFC form (canonical composition)
        # This handles accented characters properly
        text = unicodedata.normalize('NFC', text)

        # Replace Unicode quotation marks with standard ones
        text = text.replace('\u201c', '"')  # Left double quote "
        text = text.replace('\u201d', '"')  # Right double quote "
        text = text.replace('\u2018', "'")  # Left single quote '
        text = text.replace('\u2019', "'")  # Right single quote '

        # Replace Unicode dashes with standard hyphen
        text = text.replace('\u2013', '-')  # En dash –
        text = text.replace('\u2014', '-')  # Em dash —

        # Replace bullet points with standard dash
        text = text.replace('\u2022', '-')  # Bullet •
        text = text.replace('\u00b7', '-')  # Middle dot ·

        return text

    def preserve_petroleum_context(self, text):
        """
        Ensure petroleum terminology is preserved correctly.

        This function does NOT change the terms — it verifies
        that numbers stay attached to their units.

        Problem: "10.2\nppg" (split across lines) should be "10.2 ppg"
        We fix this for all petroleum units.
        """
        for unit in self.petroleum_units:
            # Pattern: number followed by newline followed by unit
            # Fix: join them with a space
            pattern = re.compile(
                r'(\d+\.?\d*)\s*\n\s*(' + re.escape(unit) + r')',
                re.IGNORECASE
            )
            text = pattern.sub(r'\1 \2', text)

        return text

    def remove_excessive_repetition(self, text):
        """
        Remove lines that repeat more than 3 times in the document.

        In some PDFs, certain headers or watermarks appear on
        every single page. After we split by page and join,
        these appear dozens of times and dominate the content.

        We identify lines that repeat more than 3 times and
        keep only the first 2 occurrences.
        """
        lines = text.split('\n')
        line_counts = {}  # Count how many times each line appears

        for line in lines:
            stripped = line.strip()
            if stripped:  # Skip empty lines
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        # Find lines that repeat more than 3 times
        # These are likely headers/footers
        repetitive_lines = {
            line for line, count in line_counts.items()
            if count > 3 and len(line) < 100
            # Only flag SHORT lines — long content lines won't be headers
        }

        # Filter: keep each repetitive line max 2 times
        kept_counts = {}
        filtered_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped in repetitive_lines:
                kept_counts[stripped] = kept_counts.get(stripped, 0) + 1
                if kept_counts[stripped] <= 2:
                    filtered_lines.append(line)
                # If we've seen it more than 2 times, skip it
            else:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def clean(self, text, document_name=""):
        """
        MAIN METHOD: Apply all cleaning steps in the correct order.

        This is the function we call from outside this class.
        It runs all cleaning steps in sequence.

        Order matters:
        1. Remove null bytes first (other operations may fail on nulls)
        2. Normalize unicode (convert special chars before regex)
        3. Fix hyphenation (before removing whitespace)
        4. Preserve petroleum terms (before any other changes)
        5. Remove headers/footers (pattern-based)
        6. Remove repetitive lines (count-based)
        7. Normalize whitespace (final cleanup)

        Returns: cleaned text string
        """
        if not text or len(text.strip()) == 0:
            return ""

        original_length = len(text)

        # Apply each cleaning step
        text = self.remove_null_bytes(text)
        text = self.normalize_unicode(text)
        text = self.fix_hyphenated_words(text)
        text = self.preserve_petroleum_context(text)
        text = self.remove_headers_footers(text)
        text = self.remove_excessive_repetition(text)
        text = self.normalize_whitespace(text)

        # Final strip to remove leading/trailing whitespace
        text = text.strip()

        cleaned_length = len(text)
        reduction = ((original_length - cleaned_length) / original_length * 100
                     if original_length > 0 else 0)

        # Log cleaning statistics
        if document_name:
            print(f"    Cleaned '{document_name}': "
                  f"{original_length:,} → {cleaned_length:,} chars "
                  f"({reduction:.1f}% noise removed)")

        return text