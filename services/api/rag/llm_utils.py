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

if LANGCHAIN_AVAILABLE and config.GOOGLE_API_KEY:
    try:
        # --- THIS BLOCK IS CORRECTED ---
        # Vector DB (Chroma)
        # Explicitly pass host and port to the client.
        chroma_client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
        chroma_client.heartbeat() # Check connection
        
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=config.GOOGLE_API_KEY)
        
        # Pass the initialized client to the Chroma vector store.
        vector_store = Chroma(
            client=chroma_client,
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=embedding_function
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        logger.info("Connected to Chroma collection '%s' at %s:%s", config.CHROMA_COLLECTION, config.CHROMA_HOST, config.CHROMA_PORT)

    except Exception as e:
        logger.warning("Chroma vector store initialization failed: %s", e)

    try:
        # LLM Chain (Gemini)
        llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0.0)
        SQL_PROMPT = """You are a strict SQL generator for an ARGO float PostgreSQL schema (tables: profiles, levels).
Rules:
- Output only a single SELECT statement without a trailing semicolon.
- Always qualify column names with their table name (e.g., profiles.juld, levels.temp_degc).
- Use `profiles.juld` for any date or time filtering.
- Use SQL aggregates (AVG, MAX, MIN, COUNT) when asked for them.
Context from schema: {context}
User Question: {question}
SQL Query:
"""
        PROMPT = PromptTemplate(template=SQL_PROMPT, input_variables=["context", "question"])
        llm_chain = LLMChain(llm=llm, prompt=PROMPT)
        logger.info("LLM chain configured with model %s", config.GEMINI_MODEL)
    except Exception as e:
        logger.exception("LLM chain setup failed: %s", e)
else:
    logger.info("LLM or Vector retrieval not configured (Langchain/Google API key missing).")

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