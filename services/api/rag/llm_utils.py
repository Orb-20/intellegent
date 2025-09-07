# services/api/rag/llm_utils.py
import logging
from typing import Any, List, Tuple, Dict

from . import config

logger = logging.getLogger(__name__)

# --- Optional LLM / vector imports (best-effort) ---
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_community.vectorstores import Chroma
    import chromadb
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# --- Initializations ---
retriever = None
llm_chain = None
answer_chain = None

# --- THIS IS THE FIX ---
# Add explicit checks and logging for dependencies and configuration.
if not config.GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY environment variable is not set. LLM features will be disabled.")
elif not LANGCHAIN_AVAILABLE:
    logger.warning("Langchain dependencies are not installed. LLM features will be disabled.")
else:
    # Vector DB Initialization
    try:
        chroma_client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
        chroma_client.heartbeat()
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config.GOOGLE_API_KEY)
        vector_store = Chroma(
            client=chroma_client,
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=embedding_function
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        logger.info("Successfully connected to Chroma collection '%s' at %s:%s", config.CHROMA_COLLECTION, config.CHROMA_HOST, config.CHROMA_PORT)
    except Exception as e:
        logger.warning("Chroma vector store initialization failed: %s", e)

    # LLM Chains Initialization
    try:
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0.0, google_api_key=config.GOOGLE_API_KEY)
        
        # 1. Chain for SQL Generation
        SQL_PROMPT_TEXT = """You are a strict SQL generator for an ARGO float PostgreSQL schema (tables: profiles, levels).
Rules:
- Output only a single SELECT statement without a trailing semicolon.
- Always qualify column names with their table name (e.g., profiles.juld, levels.temp_degc).
- Use `profiles.juld` for any date or time filtering.
- Use SQL aggregates (AVG, MAX, MIN, COUNT) when asked for them.
Context from schema: {context}
User Question: {question}
SQL Query:
"""
        SQL_PROMPT = PromptTemplate(template=SQL_PROMPT_TEXT, input_variables=["context", "question"])
        llm_chain = LLMChain(llm=llm, prompt=SQL_PROMPT)
        logger.info("LLM chain for SQL generation configured with model %s", config.GEMINI_MODEL)

        # 2. Chain for generating polished natural language answers
        ANSWER_PROMPT_TEXT = """You are FloatChat, an expert data analyst for ARGO floats.
Instructions:
- Produce a concise (1-2 sentence) answer to the user's question based on the data summary.
- Do NOT repeat the SQL query or raw diagnostics. Use them only for context.
- Suggest a single, relevant follow-up question.

User question: {question}
Data summary: {context}

Answer:"""
        ANSWER_PROMPT = PromptTemplate(template=ANSWER_PROMPT_TEXT, input_variables=["context", "question"])
        answer_chain = LLMChain(llm=llm, prompt=ANSWER_PROMPT)
        logger.info("LLM chain for answer generation configured with model %s", config.GEMINI_MODEL)

    except Exception as e:
        logger.exception("LLM chain setup failed. This is likely due to an invalid GOOGLE_API_KEY or network issues: %s", e)


def build_context_from_docs(docs: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Builds a context string and a provenance list from retrieved documents."""
    parts, prov = [], []
    if not docs:
        return '', []
    for d in docs:
        md = getattr(d, 'metadata', {})
        content = f"Column: {md.get('column')}, Table: {md.get('table')}, Description: {md.get('description')}"
        parts.append(content)
        prov.append({'source': md.get('source', md.get('table')), 'content': content})
    return '\n'.join(parts), prov