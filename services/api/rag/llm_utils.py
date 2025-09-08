# services/api/rag/llm_utils.py
"""
LLM utilities for RAG workflows — Mistral-first by default, easy switch to Gemini.

Usage:
- Prefer Mistral (default): llm_utils.init_llm() or rely on legacy aliases initialized at import.
- Force Gemini: llm_utils.init_llm(prefer="gemini") or llm_utils.set_primary_provider("gemini")
- Use wrapper functions for safe calls:
    run_intent_classification(question)
    run_sql_generation(context, question)
    run_answer_generation(context, question)
    run_sql_correction(schema, question, sql_query, error_message)
    run_plot_selection(question, columns)
"""

import logging
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple

from . import config

# Optional imports; keep module importable even if these libs are missing.
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_mistralai.chat_models import ChatMistralAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_community.vectorstores import Chroma
    import chromadb
    LANGCHAIN_AVAILABLE = True
except Exception:
    GoogleGenerativeAIEmbeddings = None  # type: ignore
    ChatGoogleGenerativeAI = None  # type: ignore
    ChatMistralAI = None  # type: ignore
    PromptTemplate = None  # type: ignore
    LLMChain = None  # type: ignore
    Chroma = None  # type: ignore
    chromadb = None  # type: ignore
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Exceptions
class LLMUtilsError(Exception):
    """Base exception for llm_utils module."""

class LLMUnavailableError(LLMUtilsError):
    """Raised when no LLM provider can be initialized."""

class VectorStoreUnavailableError(LLMUtilsError):
    """Raised when vectorstore cannot be initialized."""

# -------------------------
# Internal state (lazy)
# -------------------------
_retriever: Optional[Any] = None
_llm: Optional[Any] = None
_chroma_client: Optional[Any] = None
_chroma_store: Optional[Any] = None
_chains: Dict[str, Any] = {}
_llm_provider: Optional[str] = None  # 'mistral' or 'gemini'
_preferred_provider: str = "mistral"  # default preference; can be changed via set_primary_provider()

# Backward-compatible exports (populated by _init_legacy_chains)
intent_chain: Optional[Any] = None
llm_chain: Optional[Any] = None
answer_chain: Optional[Any] = None
correction_chain: Optional[Any] = None
plot_selection_chain: Optional[Any] = None
retriever: Optional[Any] = None
llm: Optional[Any] = None

# -------------------------
# Prompt templates
# -------------------------
_INTENT_PROMPT = """Your job is to classify the user's intent. Respond with a single word from the options provided.
Options:
- data_query: The user is asking a question about ARGO data, requesting a calculation, or asking to find profiles.
- greeting: The user is saying hi, hello, or a similar greeting.
- unknown: The user's intent is unclear or not related to data.

User input: {question}
Intent:"""

_SQL_PROMPT = """You are an expert PostgreSQL data analyst for the ARGO oceanographic database. Your task is to convert a user's question into a single, precise, and executable SQL query.

**Database Schema:**
- `profiles` (profile_id, platform_number, cycle_number, juld, latitude, longitude, project_name)
- `levels` (level_id, profile_id, pres_dbar, temp_degc, psal_psu)

Instructions:
1. Analyze the request: aggregation, distribution, comparison, timeseries, profile, etc.
2. Use context from domain knowledge to interpret terms.
3. Write a short plan (Goal, Columns, Joins, Filters, Aggregation).
4. Output only the final SQL in a single code block.

Context:
{context}

User question:
{question}

Plan:
1. Goal:
2. Columns:
3. Joins:
4. Filters:
5. Aggregation:

Final SQL Query:
```sql
(Your SQL query here)
```"""

_ANSWER_PROMPT = """You are FloatChat, a helpful assistant for ARGO ocean data. Based on a short data summary, provide a concise 1-2 sentence answer and suggest a relevant follow-up.

User question: {question}
Data summary: {context}
Answer:"""

_CORRECTION_PROMPT = """You are an expert PostgreSQL debugger. The user's query failed. Fix the SQL query based on the error.
Rules: output only the corrected SQL query (no explanation).

PostgreSQL Schema: {schema}
Original Question: {question}
Faulty SQL: {sql_query}
Database Error: {error_message}
Corrected SQL Query:"""

_PLOT_PROMPT = """You are a data visualization expert. Return a single clean JSON object describing the plot to generate.

User Question: {question}
Available Data Columns: {columns}

Rules:
- "plot_type" must be one of ["histogram","bar_chart","scatter_geo","timeseries","scatter","profile","none"].
- For "profile", y-axis MUST be pressure (e.g., pres_dbar).
- Provide title and axis fields where applicable.

Output:
"""

# -------------------------
# Utilities
# -------------------------
def _extract_first_json(s: str) -> Optional[Dict[str, Any]]:
    """
    Try robustly to extract the first JSON object from a string.
    Strategy: find the first '{' and try progressively larger substrings ending at later '}' chars.
    This is resilient to markdown/code fences and surrounding text.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    start = s.find("{")
    if start == -1:
        return None
    # Try expanding to later closing braces
    for end in range(len(s) - 1, start - 1, -1):
        if s[end] != "}":
            continue
        candidate = s[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            continue
    # Fallback: look for simple { ... } via regex
    try:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None

def _safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    """Wrapper that tries several strategies to parse JSON found inside LLM responses."""
    try:
        return _extract_first_json(s)
    except Exception:
        try:
            return json.loads(s)
        except Exception:
            return None

def _is_rate_limit_error(exc: Exception, resp_text: Optional[str] = None) -> bool:
    """
    Detect probable quota / rate limit errors from exception message or response text.
    Looks for '429', 'quota', 'rate limit', 'too many requests', or the Gemini quota metric.
    """
    text = ""
    try:
        text = str(exc) if exc is not None else ""
    except Exception:
        text = ""
    if resp_text:
        text += " " + resp_text
    text = text.lower()
    indicators = [
        "429", "quota", "rate limit", "too many requests", "rate_limited",
        "generativelanguage.googleapis.com/generate_content_free_tier_requests",
    ]
    return any(ind in text for ind in indicators)

def _safe_prompt(template: str, **kwargs) -> str:
    try:
        return template.format(**kwargs)
    except Exception:
        out = template
        for k, v in kwargs.items():
            out = out.replace("{" + k + "}", str(v))
        return out

# -------------------------
# Vectorstore initialization (optional)
# -------------------------
def init_vectorstore(max_retries: int = 3, retry_delay: float = 1.0) -> Optional[Any]:
    """
    Initialize or return cached Chroma retriever. Requires config.CHROMA_HOST, CHROMA_PORT, CHROMA_COLLECTION and GOOGLE_API_KEY.
    If those aren't set, vectorstore is skipped and None is returned.
    """
    global _retriever, _chroma_client, _chroma_store, retriever
    if _retriever is not None:
        return _retriever

    if not LANGCHAIN_AVAILABLE:
        logger.warning("LangChain packages not installed; skipping vectorstore.")
        return None

    host = getattr(config, "CHROMA_HOST", None)
    port = getattr(config, "CHROMA_PORT", None)
    collection = getattr(config, "CHROMA_COLLECTION", None)
    google_key = getattr(config, "GOOGLE_API_KEY", None)
    if not (host and port and collection and google_key):
        logger.info("Vectorstore config (CHROMA_HOST/PORT/COLLECTION/GOOGLE_API_KEY) incomplete; skipping vectorstore.")
        return None

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            _chroma_client = chromadb.HttpClient(host=host, port=port)
            _chroma_client.heartbeat()
            embedding_func = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_key)
            _chroma_store = Chroma(client=_chroma_client, collection_name=collection, embedding_function=embedding_func)
            _retriever = _chroma_store.as_retriever(search_kwargs={"k": 5})
            retriever = _retriever
            logger.info("Connected to Chroma collection '%s' at %s:%s", collection, host, port)
            return _retriever
        except Exception as exc:
            logger.warning("Chroma init attempt %d failed: %s", attempt, exc)
            time.sleep(retry_delay * attempt)
    logger.error("Chroma initialization failed after %d attempts.", max_retries)
    return None

# -------------------------
# LLM initialization — Mistral-first by default
# -------------------------
def init_llm(prefer: Optional[str] = None, allow_fallback: bool = True) -> Any:
    """
    Initialize and cache an LLM provider.

    - prefer: "mistral" or "gemini" (if None, uses module-level _preferred_provider)
    - allow_fallback: if True, will try the other provider if the preferred one can't be initialized

    Returns the initialized LLM object. Raises LLMUnavailableError if no provider works.
    """
    global _llm, _llm_provider, llm

    if _llm is not None and _llm_provider is not None:
        # If prefer provided and differs, allow re-init
        chosen = prefer.lower() if prefer else _preferred_provider
        if chosen == _llm_provider:
            return _llm
        # else fall through to reinit with preference

    if not LANGCHAIN_AVAILABLE:
        logger.error("LangChain packages not available; cannot initialize LLM.")
        raise LLMUnavailableError("LangChain not installed")

    # Determine preference
    pref = (prefer or _preferred_provider or "mistral").lower()

    def try_mistral() -> Optional[Any]:
        key = getattr(config, "MISTRAL_API_KEY", None)
        model = getattr(config, "MISTRAL_MODEL", None)
        if key and model and ChatMistralAI is not None:
            try:
                obj = ChatMistralAI(model=model, temperature=0.0, api_key=key)
                logger.info("Mistral initialized: %s", model)
                return obj
            except Exception as e:
                logger.warning("Mistral init failed: %s", e)
        else:
            logger.debug("Mistral config missing or ChatMistralAI not available.")
        return None

    def try_gemini() -> Optional[Any]:
        key = getattr(config, "GOOGLE_API_KEY", None)
        model = getattr(config, "GEMINI_MODEL", None)
        if key and model and ChatGoogleGenerativeAI is not None:
            try:
                obj = ChatGoogleGenerativeAI(model=model, temperature=0.0, google_api_key=key)
                logger.info("Gemini initialized: %s", model)
                return obj
            except Exception as e:
                logger.warning("Gemini init failed: %s", e)
        else:
            logger.debug("Gemini config missing or ChatGoogleGenerativeAI not available.")
        return None

    # Preference logic
    providers = []
    if pref == "mistral":
        providers = [try_mistral, try_gemini] if allow_fallback else [try_mistral]
    elif pref == "gemini":
        providers = [try_gemini, try_mistral] if allow_fallback else [try_gemini]
    else:
        providers = [try_mistral, try_gemini] if allow_fallback else [try_mistral]

    for try_fn in providers:
        obj = try_fn()
        if obj is not None:
            _llm = obj
            llm = _llm
            # determine provider name
            if try_fn is try_mistral:
                _llm_provider = "mistral"
            else:
                _llm_provider = "gemini"
            logger.info("LLM provider set to: %s", _llm_provider)
            return _llm

    logger.error("No LLM providers could be initialized (checked Mistral and Gemini).")
    raise LLMUnavailableError("No LLM providers available.")

def set_primary_provider(provider: str, allow_fallback: bool = True) -> Any:
    """
    Convenience to set preferred provider at runtime. provider should be "mistral" or "gemini".
    After calling this, chains will be reinitialized to bind to the chosen provider.
    """
    global _preferred_provider
    _preferred_provider = provider.lower()
    # Reinitialize llm/chains using new preference
    init_llm(prefer=_preferred_provider, allow_fallback=allow_fallback)
    init_chains(force=True)
    return _llm_provider

# -------------------------
# Chains initialization (lazy)
# -------------------------
def init_chains(force: bool = False) -> Dict[str, Any]:
    """
    Initialize/cached LLM chains (intent, sql, answer, correction, plot).
    Returns the dictionary of chains.
    """
    global _chains, intent_chain, llm_chain, answer_chain, correction_chain, plot_selection_chain

    if _chains and not force:
        return _chains

    try:
        llm_local = init_llm(prefer=_preferred_provider, allow_fallback=True)
    except Exception:
        logger.exception("LLM initialization failed; chains unavailable.")
        return {}

    def _make_chain(prompt_text: str, input_vars: List[str]) -> Optional[Any]:
        try:
            tmpl = PromptTemplate(template=prompt_text, input_variables=input_vars)
            return LLMChain(llm=llm_local, prompt=tmpl)
        except Exception:
            logger.exception("Failed to create chain for prompt.", exc_info=True)
            return None

    _chains = {
        "intent": _make_chain(_INTENT_PROMPT, ["question"]),
        "sql": _make_chain(_SQL_PROMPT, ["context", "question"]),
        "answer": _make_chain(_ANSWER_PROMPT, ["context", "question"]),
        "correction": _make_chain(_CORRECTION_PROMPT, ["schema", "question", "sql_query", "error_message"]),
        "plot": _make_chain(_PLOT_PROMPT, ["question", "columns"]),
    }

    # Backward-compatible aliases
    intent_chain = _chains.get("intent")
    llm_chain = _chains.get("sql")  # legacy naming
    answer_chain = _chains.get("answer")
    correction_chain = _chains.get("correction")
    plot_selection_chain = _chains.get("plot")

    logger.info("LLM chains initialized: %s", ", ".join([k for k, v in _chains.items() if v]))
    return _chains

# -------------------------
# Chain-run wrapper with optional provider failover
# -------------------------
def _run_chain(chain_name: str, inputs: Dict[str, Any], allow_failover: bool = True) -> Optional[str]:
    """
    Run a named chain safely.
    If a rate-limit/quota error is detected and allow_failover is True,
    will try to reinitialize with the alternate provider and retry once.
    """
    global _llm_provider
    chains = init_chains()
    chain = chains.get(chain_name)
    if chain is None:
        logger.warning("Requested chain '%s' not available.", chain_name)
        return None

    attempts = 0
    max_attempts = 2  # initial attempt + one retry after failover
    last_exc = None

    while attempts < max_attempts:
        attempts += 1
        try:
            if hasattr(chain, "run"):
                result = chain.run(inputs)
            else:
                result = chain(inputs)  # type: ignore

            if result is None:
                logger.debug("Chain '%s' returned None.", chain_name)
                return None
            if isinstance(result, (dict, list)):
                return json.dumps(result)
            return str(result)
        except Exception as exc:
            last_exc = exc
            logger.exception("Chain '%s' raised exception on attempt %d: %s", chain_name, attempts, exc)
            # If this looks like rate-limit/quota, try failover
            if allow_failover and _is_rate_limit_error(exc):
                logger.warning("Detected quota/rate limit from provider '%s'. Attempting provider switch.", _llm_provider)
                alt = "mistral" if _llm_provider == "gemini" else "gemini"
                try:
                    init_llm(prefer=alt, allow_fallback=True)
                    init_chains(force=True)
                    chain = _chains.get(chain_name)
                    if chain is None:
                        logger.error("After provider switch, chain '%s' unavailable.", chain_name)
                        break
                    # retry will occur
                    continue
                except Exception as rexc:
                    logger.exception("Provider failover failed: %s", rexc)
                    break
            break

    logger.error("Chain '%s' failed after %d attempts. Last exception: %s", chain_name, attempts, last_exc)
    return None

# -------------------------
# High-level runners (use these in your app)
# -------------------------
def run_intent_classification(question: str) -> Optional[str]:
    resp = _run_chain("intent", {"question": question})
    if resp:
        token = resp.strip().splitlines()[0].strip()
        token = re.sub(r'[^a-zA-Z0-9_]', '', token).lower()
        return token or None
    return None

def run_sql_generation(context: str, question: str) -> Optional[str]:
    resp = _run_chain("sql", {"context": context or "", "question": question})
    if not resp:
        return None
    m = re.search(r"```sql\s*(.*?)```", resp, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```\s*(.*?)```", resp, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return resp.strip()

def run_answer_generation(context: str, question: str) -> Optional[str]:
    resp = _run_chain("answer", {"context": context or "", "question": question})
    if resp:
        return resp.strip()
    return None

def run_sql_correction(schema: str, question: str, sql_query: str, error_message: str) -> Optional[str]:
    resp = _run_chain("correction", {"schema": schema, "question": question, "sql_query": sql_query, "error_message": error_message})
    if not resp:
        return None
    m = re.search(r"```sql\s*(.*?)```", resp, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```\s*(.*?)```", resp, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return resp.strip()

def run_plot_selection(question: str, columns: List[str]) -> Optional[Dict[str, Any]]:
    cols_repr = json.dumps(columns)
    resp = _run_chain("plot", {"question": question, "columns": cols_repr})
    if not resp:
        return None
    parsed = _safe_json_loads(resp)
    if parsed:
        return parsed
    # permissive fallback: parse key:value lines
    try:
        d = {}
        for line in resp.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
        if d:
            return d
    except Exception:
        logger.debug("Could not permissively parse plot selection.", exc_info=True)
    return None

# -------------------------
# Context builder
# -------------------------
def build_context_from_docs(docs: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    if not docs:
        return "", []
    parts: List[str] = []
    provenance: List[Dict[str, Any]] = []
    for doc in docs:
        try:
            if hasattr(doc, "metadata"):
                source = getattr(doc, "metadata", {}).get("source")
            elif isinstance(doc, dict):
                source = doc.get("metadata", {}).get("source") if doc.get("metadata") else None
            else:
                source = None
            content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else None) or str(doc)
            parts.append(f"--- Knowledge from {source or 'unknown'} ---\n{content}")
            provenance.append({"source": source, "content": content})
        except Exception:
            logger.debug("Skipping malformed doc in build_context_from_docs.", exc_info=True)
            continue
    return "\n\n".join(parts), provenance

# -------------------------
# Legacy initialization (best-effort on import) — prefers Mistral
# -------------------------
def _init_legacy_chains():
    """
    Best-effort initialization to preserve legacy usage patterns.
    This will prefer Mistral (module-level preference) but will not raise on failure.
    """
    global intent_chain, llm_chain, answer_chain, correction_chain, plot_selection_chain, retriever, llm
    try:
        init_vectorstore()
    except Exception:
        logger.debug("Vectorstore init skipped or failed during legacy init.", exc_info=True)
    try:
        init_chains()
        intent_chain = _chains.get("intent")
        llm_chain = _chains.get("sql")
        answer_chain = _chains.get("answer")
        correction_chain = _chains.get("correction")
        plot_selection_chain = _chains.get("plot")
    except Exception:
        logger.debug("Chains init skipped or failed during legacy init.", exc_info=True)
    try:
        retriever = _retriever
        llm = _llm
    except Exception:
        pass

# Run legacy init at import time (best-effort). This will prefer Mistral.
try:
    _init_legacy_chains()
except Exception:
    logger.debug("Legacy init encountered an error; continuing.", exc_info=True)

# -------------------------
# Public exports
# -------------------------
__all__ = [
    "init_vectorstore",
    "init_llm",
    "init_chains",
    "set_primary_provider",
    "run_intent_classification",
    "run_sql_generation",
    "run_answer_generation",
    "run_sql_correction",
    "run_plot_selection",
    "build_context_from_docs",
    # legacy globals
    "intent_chain",
    "llm_chain",
    "answer_chain",
    "correction_chain",
    "plot_selection_chain",
    "retriever",
    "llm",
    # exceptions
    "LLMUtilsError",
    "LLMUnavailableError",
    "VectorStoreUnavailableError",
]
