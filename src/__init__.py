# ============================================================
#
# This file makes the src/ folder a Python package.
# Without this file, Python cannot import from src/.
#
# It also provides convenient imports so other files
# can write: from src import DocumentExtractor
# instead of: from src.pdf_extractor import DocumentExtractor
# ============================================================

from src.text_cleaner import TextCleaner
from src.pdf_extractor import DocumentExtractor
from src.chunker import SmartChunker
from src.memory import ConversationMemory

# Define what gets exported when someone does: from src import *
__all__ = [
    'TextCleaner',
    'DocumentExtractor',
    'SmartChunker',
    'ConversationMemory'
]