# ============================================================
#
# The core Q&A pipeline — combines retrieval + generation.
#
# This is the "brain" of the entire system.
# It orchestrates:
# 1. Retrieving relevant chunks from ChromaDB
# 2. Building a precise prompt for the LLM
# 3. Calling the LLM via HuggingFace API
# 4. Returning the answer with source citations
# ============================================================

import os
from typing import List, Dict, Optional
from src.embeddings import VectorStoreManager
from src.memory import ConversationMemory

# Load API token from environment
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")

# LLM model to use
# Mistral-7B-Instruct: Best free model for technical Q&A
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# LLM parameters
MAX_NEW_TOKENS = 512    # Maximum length of generated answer
TEMPERATURE = 0.1       # Very low = factual, deterministic answers
                        # 0.0 = fully deterministic
                        # 1.0 = creative/random (bad for facts)

# How many chunks to retrieve for each question
TOP_K_CHUNKS = 4


class DrillingReportQA:
    """
    Complete Q&A system for petroleum drilling reports.

    This class is the main interface between:
    - The user (asking questions)
    - ChromaDB (storing document knowledge)
    - The LLM (generating answers)

    Usage:
        qa = DrillingReportQA()
        result = qa.ask("Was there a stuck pipe incident?")
        print(result['answer'])
    """

    def __init__(self):
        """
        Initialize all components of the Q&A system.
        """
        print("Initializing Q&A System...")

        # Initialize vector store (ChromaDB + embedding model)
        self.vector_store = VectorStoreManager()

        # Initialize conversation memory (SQLite)
        self.memory = ConversationMemory()

        # Initialize LLM client
        self.llm_client = self._initialize_llm()

        print("   Q&A System ready\n")

    def _initialize_llm(self):
        """
        Initialize the HuggingFace LLM client.

        We use the HuggingFace Inference API which:
        - Is free for reasonable usage
        - Requires only our API token
        - Runs the model on HuggingFace servers (no GPU needed locally)
        """
        from huggingface_hub import InferenceClient

        if not HUGGINGFACE_TOKEN:
            raise ValueError(
                "HuggingFace API token not found.\n"
                "Add HUGGINGFACEHUB_API_TOKEN to your .env file"
            )

        # InferenceClient: HuggingFace's official API client
        client = InferenceClient(
            model=LLM_MODEL,
            token=HUGGINGFACE_TOKEN
        )

        print(f"   LLM initialized: {LLM_MODEL}")
        return client

    def build_prompt(self, question: str,
                     context_chunks: List[Dict]) -> str:
        """
        Build a carefully engineered prompt for the LLM.

        Prompt engineering is the art of writing instructions
        that make the LLM give better answers.

        Our prompt has three parts:
        1. SYSTEM: Who the LLM is and what rules it follows
        2. CONTEXT: The relevant document chunks
        3. QUESTION: What the user asked

        Why this structure works:
        - Clear role → LLM uses petroleum vocabulary correctly
        - Explicit constraints → Prevents hallucination
        - Numbered sources → Enables citation tracking
        - Answer format → Structured, professional output
        """
        # Build the context section from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            context_parts.append(
                f"[SOURCE {i} — {chunk['document']}]\n"
                f"{chunk['text']}"
            )

        context_text = "\n\n---\n\n".join(context_parts)

        # Build the complete prompt
        # We use Mistral's instruction format: [INST] ... [/INST]
        prompt = f"""[INST] You are an expert petroleum engineer with 20 years 
of experience in drilling operations. You are analyzing Daily Drilling Reports 
(DDRs) and petroleum documents to answer questions precisely.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using information from the provided sources below
2. If the answer is not in the sources, respond exactly with:
   "This information is not found in the provided reports."
3. Always cite which source(s) you used: (Source 1), (Source 2), etc.
4. Include specific details: depths in feet, dates, measurements with units
5. Be concise but complete — include all relevant technical details
6. Do not add information from your training data

PETROLEUM DOCUMENT SOURCES:
{context_text}

ENGINEER'S QUESTION:
{question}

ANSWER (cite sources, include specific technical details): [/INST]"""

        return prompt

    def call_llm(self, prompt: str) -> str:
        """
        Send the prompt to Mistral-7B and get the answer.

        Uses HuggingFace Inference API — runs on their servers.
        Response time: typically 3-15 seconds depending on load.
        """
        try:
            # text_generation: The API call to generate text
            response = self.llm_client.text_generation(
                prompt=prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                # repetition_penalty: Prevents the LLM from repeating
                # the same phrases over and over
                repetition_penalty=1.1,
                # do_sample=False with temperature<0.2 gives
                # more deterministic, factual outputs
                do_sample=True,
                # return_full_text=False: Return only the new generated
                # text, not the original prompt repeated back
                return_full_text=False
            )

            # Clean up the response
            answer = response.strip()

            # Remove any [/INST] tags if they appear in output
            answer = answer.replace('[/INST]', '').strip()

            return answer

        except Exception as e:
            error_msg = str(e)

            # Provide helpful error messages for common failures
            if "401" in error_msg:
                return (" Authentication error: Your HuggingFace token "
                        "is invalid. Check your .env file.")
            elif "503" in error_msg:
                return (" Model is loading on HuggingFace servers. "
                        "Please try again in 30 seconds.")
            elif "429" in error_msg:
                return ("  Rate limit reached. Please wait 1 minute "
                        "before asking another question.")
            else:
                return f" LLM Error: {error_msg[:200]}"

    def ask(self, question: str,
            session_id: Optional[str] = None,
            search_all: bool = True,
            specific_collection: Optional[str] = None) -> Dict:
        """
        MAIN METHOD: Answer a question about the drilling reports.

        This orchestrates the complete RAG pipeline:
        1. Retrieve relevant chunks from ChromaDB
        2. Build an engineered prompt
        3. Call the LLM
        4. Save to conversation history
        5. Return answer with sources

        Parameters:
        - question: The user's question
        - session_id: Current session (for saving to memory)
        - search_all: If True, search all documents
                      If False, search only specific_collection
        - specific_collection: Document to search (if search_all=False)

        Returns dict with:
        - answer: The generated answer text
        - sources: List of chunks used to generate the answer
        - question: The original question (for display)
        - retrieved_chunks: Number of chunks retrieved
        """
        print(f"\n{'─'*50}")
        print(f"Question: {question}")

        # STEP 1: Retrieve relevant chunks
        print(f"Step 1: Searching ChromaDB...")

        if search_all:
            # Search across ALL indexed documents
            chunks = self.vector_store.search_all_collections(
                query=question,
                top_k=TOP_K_CHUNKS
            )
        else:
            # Search only the specified document
            chunks = self.vector_store.search(
                query=question,
                collection_name=specific_collection,
                top_k=TOP_K_CHUNKS
            )

        if not chunks:
            answer = ("No indexed documents found. Please upload and "
                      "index a drilling report first.")
            return {
                'question': question,
                'answer': answer,
                'sources': [],
                'retrieved_chunks': 0
            }

        print(f"  Retrieved {len(chunks)} relevant chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  #{i} [{chunk['document']}] "
                  f"relevance: {chunk['relevance_pct']}%")

        # STEP 2: Build prompt
        print(f"Step 2: Building prompt...")
        prompt = self.build_prompt(question, chunks)

        # STEP 3: Call LLM
        print(f"Step 3: Calling Mistral-7B...")
        answer = self.call_llm(prompt)
        print(f"  Answer generated ({len(answer)} chars)")

        # STEP 4: Save to conversation memory
        if session_id:
            source_texts = [chunk['text'][:200] for chunk in chunks]
            self.memory.save_qa_pair(
                session_id=session_id,
                question=question,
                answer=answer,
                source_chunks=source_texts
            )
            print(f"  Saved to conversation history")

        # STEP 5: Format and return result
        result = {
            'question': question,
            'answer': answer,
            'sources': chunks,
            'retrieved_chunks': len(chunks)
        }

        print(f"\n{'─'*50}")
        print(f"ANSWER:\n{answer}")
        print(f"{'─'*50}\n")

        return result

    def ask_with_history(self, question: str,
                         session_id: str,
                         conversation_history: List[Dict]) -> Dict:
        """
        Answer a question with awareness of previous questions.

        This handles follow-up questions like:
        "What caused it?" (after asking about stuck pipe)
        "How long did it last?" (referring to previous incident)

        We inject the last 3 Q&A pairs into the prompt so the
        LLM understands the conversation context.
        """
        # Build conversation context from last 3 exchanges
        history_context = ""

        if conversation_history:
            # Take last 3 exchanges (not too many — saves tokens)
            recent = conversation_history[-3:]
            history_lines = []

            for exchange in recent:
                history_lines.append(
                    f"Previous Q: {exchange['question']}\n"
                    f"Previous A: {exchange['answer'][:200]}..."
                )

            history_context = (
                "\n\nPREVIOUS CONVERSATION CONTEXT:\n" +
                "\n\n".join(history_lines) +
                "\n\nCurrent question refers to the above context."
            )

        # Add history context to the question
        enriched_question = question + history_context

        return self.ask(
            question=enriched_question,
            session_id=session_id,
            search_all=True
        )