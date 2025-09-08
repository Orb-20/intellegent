# services/api/rag/llm_utils.py
import logging
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from . import config

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

_llm: Optional[Any] = None
_chains: Dict[str, Any] = {}

_SQL_PROMPT_TEMPLATE = """You are an expert PostgreSQL data analyst for the ARGO oceanographic database. Your task is to convert a user's question into a single, precise, and executable SQL query.

**Database Schema:**
- `profiles` table: Contains metadata for each float profile (profile_id, platform_number, juld, latitude, longitude).
- `levels` table: Contains the measurements at different depths for each profile (profile_id, pres_dbar, temp_degc, psal_psu).

**CRITICAL RULES:**
1.  **JOIN Correctly:** When a query needs both location/time (from `profiles`) and measurements (from `levels`), you MUST `JOIN levels ON profiles.profile_id = levels.profile_id`.
2.  **Handle "Surface" Measurements:** When a user asks for "surface" temperature or salinity, this means a pressure less than 10 dbar. You MUST add the condition `WHERE levels.pres_dbar < 10`.
3.  **NO LIMIT on Analytical Queries:** For any question involving aggregations (AVG, COUNT, etc.) or asking for a plot/graph, you MUST NOT use a `LIMIT` clause. The system will handle aggregation.
4.  **Output ONLY SQL:** Your final output must be ONLY the SQL query, with no explanations, comments, or markdown.

**EXAMPLES (Learn from these):**
---
**User Question:** what was the average surface temperature in jan 2022?
**Context:** The user is asking for an aggregation (average) of a "surface" measurement over a time range.
**Correct SQL:**
SELECT AVG(levels.temp_degc) FROM profiles JOIN levels ON profiles.profile_id = levels.profile_id WHERE levels.pres_dbar < 10 AND profiles.juld BETWEEN '2022-01-01' AND '2022-01-31'
---
**User Question:** plot temperature vs depth for profile id 15
**Context:** The user wants all measurements for a single, specific profile.
**Correct SQL:**
SELECT levels.pres_dbar, levels.temp_degc FROM levels WHERE levels.profile_id = 15 ORDER BY levels.pres_dbar
---

**User Question:**
{question}

**Context from knowledge base:**
{context}

**Final SQL Query:**
"""

_SIMPLE_SQL_PROMPT_TEMPLATE = """Create a PostgreSQL query from the user's question. Output ONLY the SQL.
Schema: profiles(profile_id, juld, latitude, longitude), levels(profile_id, pres_dbar, temp_degc, psal_psu).
User Question: {question}
SQL:
"""

_ANSWER_PROMPT_TEMPLATE = """You are an expert assistant for ARGO ocean data. Based on the user's question and a summary of the data found, provide a concise, natural language answer.
User Question: {question}
Data Summary: {context}
Answer:
"""

_TITLE_PROMPT_TEMPLATE = """Create a short, descriptive title for a chart based on the user's question. Output only the title.
User Question: {question}
Chart Title:
"""

def init_llm_and_chains():
    global _llm, _chains
    if _llm and _chains:
        return

    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed.")

    try:
        _llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0.0, google_api_key=config.GOOGLE_API_KEY)
        _chains = {
            "sql": LLMChain(llm=_llm, prompt=PromptTemplate(template=_SQL_PROMPT_TEMPLATE, input_variables=["context", "question"])),
            "simple_sql": LLMChain(llm=_llm, prompt=PromptTemplate(template=_SIMPLE_SQL_PROMPT_TEMPLATE, input_variables=["question"])),
            "answer": LLMChain(llm=_llm, prompt=PromptTemplate(template=_ANSWER_PROMPT_TEMPLATE, input_variables=["context", "question"])),
            "plot_title": LLMChain(llm=_llm, prompt=PromptTemplate(template=_TITLE_PROMPT_TEMPLATE, input_variables=["question"])),
        }
        logger.info("LLM and chains initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize LLM or chains: %s", e)
        _llm, _chains = None, {}

def _run_chain(chain_name: str, **kwargs) -> Optional[str]:
    init_llm_and_chains()
    if chain_name not in _chains:
        return None
    try:
        response = _chains[chain_name].run(kwargs)
        return response.strip()
    except Exception as e:
        logger.exception("Chain '%s' failed to run: %s", chain_name, e)
        return None

def run_sql_generation(context: str, question: str) -> Optional[str]:
    return _run_chain("sql", context=context, question=question)

def run_simple_sql_generation(question: str) -> Optional[str]:
    return _run_chain("simple_sql", question=question)

def run_answer_generation(context: str, question: str) -> Optional[str]:
    return _run_chain("answer", context=context, question=question)

def run_plot_title_generation(question: str) -> Optional[str]:
    return _run_chain("plot_title", question=question)