#  Drilling Report Q&A Bot
### AI-Powered Document Intelligence for the Petroleum Industry

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue.svg)](https://www.docker.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green.svg)](https://langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Persistent-orange.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io/)

---

##  Project Overview

An intelligent Q&A system that allows petroleum engineers to upload 
Daily Drilling Reports (DDRs) and instantly get precise answers to 
technical questions — instead of manually reading through hundreds 
of pages of reports.

Built using **Retrieval-Augmented Generation (RAG)** architecture 
with real Equinor Volve field data.

---

##  Problem Statement

Oil drilling operations generate hundreds of pages of Daily Drilling 
Reports (DDRs) every week. Engineers spend 30-40% of their time 
searching for specific information across these documents. Missed 
information can lead to repeated incidents, suboptimal decisions, 
and significant financial losses ($50,000-$500,000/day in rig costs).

**This system reduces report search time from hours to seconds.**

---

##  Key Features

-  **Dual PDF Support** — Handles both digital and scanned PDFs
  - Digital PDFs: Direct text extraction via `pdfplumber`
  - Scanned PDFs: OCR via `Tesseract` engine
-  **Semantic Search** — Finds relevant content by meaning, not keywords
-  **Persistent Memory** — ChromaDB saves document index between sessions
-  **Conversation History** — SQLite saves all Q&A sessions
-  **Batch Processing** — Handles large documents (194+ pages) safely
-  **Containerized** — Docker ensures consistent deployment anywhere

---

##  System Architecture