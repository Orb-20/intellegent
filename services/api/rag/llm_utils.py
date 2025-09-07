# services/api/rag/llm_utils.py
import logging
from typing import Any, List, Tuple, Dict

from . import config

# Langchain imports with community versions for compatibility
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_community.vectorstores import Chroma
    import chromadb
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Initializations ---
retriever = None
llm_chain = None
answer_chain = None
correction_chain = None
intent_chain = None
plot_selection_chain = None

if not config.GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY environment variable is not set. LLM features will be disabled.")
elif not LANGCHAIN_AVAILABLE:
    logger.warning("Langchain dependencies are not installed. LLM features will be disabled.")
else:
    # --- Vector DB Initialization ---
    try:
        chroma_client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
        chroma_client.heartbeat()
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=config.GOOGLE_API_KEY
        )
        vector_store = Chroma(
            client=chroma_client,
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=embedding_function
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        logger.info(
            "Successfully connected to Chroma collection '%s' at %s:%s",
            config.CHROMA_COLLECTION, config.CHROMA_HOST, config.CHROMA_PORT
        )
    except Exception as e:
        logger.warning("Chroma vector store initialization failed: %s", e)

    # --- LLM Chains Initialization ---
    try:
        intent_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.0,
            google_api_key=config.GOOGLE_API_KEY
        )
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=0.0,
            google_api_key=config.GOOGLE_API_KEY
        )

        # 0. Intent Classification Chain
        INTENT_PROMPT_TEXT = """Your job is to classify the user's intent. Respond with a single word from the options provided.
Options:
- data_query: The user is asking a question about ARGO data, requesting a calculation, or asking to find profiles.
- greeting: The user is saying hi, hello, or a similar greeting.
- unknown: The user's intent is unclear or not related to data.

User input: {question}
Intent:"""
        INTENT_PROMPT = PromptTemplate(template=INTENT_PROMPT_TEXT, input_variables=["question"])
        intent_chain = LLMChain(llm=intent_llm, prompt=INTENT_PROMPT)
        logger.info("LLM chain for Intent Classification configured.")

        # 1. Expert-Level SQL Generation Chain (Upgraded)
        SQL_PROMPT_TEXT = """You are an expert PostgreSQL data analyst for the ARGO oceanographic database. Your task is to convert a user's question into a single, precise, and executable SQL query.

**Database Schema:**
- `profiles` (profile_id, platform_number, cycle_number, juld, latitude, longitude, project_name)
- `levels` (level_id, profile_id, pres_dbar, temp_degc, psal_psu)

**Instructions:**
1.  **Analyze the Request:** First, deeply understand the user's goal. Are they asking for an aggregation (AVG, COUNT), a comparison, a distribution, or a direct plot of variables?
2.  **Consult Your Knowledge:** You have been provided with critical "Context from domain knowledge". You MUST use this context to correctly interpret specialized terms (like 'surface temperature'), geographic locations, and date logic.
3.  **Formulate a Plan (Chain of Thought):** Before writing the query, you MUST articulate your plan.
    - **Goal:** What is the final data the user wants to see?
    - **Necessary Columns:** Which columns are required for the SELECT, WHERE, and GROUP BY clauses? (e.g., for a temperature distribution, you need `levels.temp_degc`).
    - **Joins:** Do `profiles` and `levels` need to be joined? (Almost always, if you need both location/date and measurements).
    - **Filtering (WHERE):** What conditions are needed?
    - **Aggregation (GROUP BY):** Is the user asking to count or average "per" category? If so, you MUST use `COUNT()` or `AVG()` and a `GROUP BY` clause.
    - **CRITICAL:** Do NOT just `SELECT * FROM profiles LIMIT 200`. This is a lazy and incorrect default. Always select the specific columns the user needs.
4.  **Write the Final SQL:** Based on your plan, write the final, single, executable SQL query. Do not output anything else.

**Context from domain knowledge:**
{context}

**User Question:**
{question}

**Chain of Thought:**
1.  **Goal:**
2.  **Necessary Columns:**
3.  **Joins:**
4.  **Filtering (WHERE):**
5.  **Aggregation (GROUP BY):**

**Final SQL Query:**
```sql
(Your SQL query here)
```"""
        SQL_PROMPT = PromptTemplate(template=SQL_PROMPT_TEXT, input_variables=["context", "question"])
        llm_chain = LLMChain(llm=llm, prompt=SQL_PROMPT)
        logger.info("LLM chain for SQL generation configured with Chain-of-Thought prompt.")

        # 2. Natural Language Answer Chain
        ANSWER_PROMPT_TEXT = """You are FloatChat, a helpful AI assistant for exploring ARGO ocean data. Provide a clear, natural language answer to the user's question based on the provided data summary.
Instructions:
- If the summary contains a direct answer (e.g., a single aggregate value), state it clearly.
- If the summary indicates a table of results, state how many items were found.
- If no results were found, say so gracefully.
- Keep the answer to 1-2 sentences and suggest a relevant follow-up question.
Example 1: User question: What is the average salinity? Data summary: Query returned 1 row. The result is a single value: 34.8. Answer: The average salinity was 34.8 PSU. Would you like to see the salinity variation over the month?
Example 2: User question: List profiles. Data summary: Query returned 87 rows. Answer: I found 87 profiles. Would you like to narrow down the date range?
Your Turn:
User question: {question}
Data summary: {context}
Answer:"""
        ANSWER_PROMPT = PromptTemplate(template=ANSWER_PROMPT_TEXT, input_variables=["context", "question"])
        answer_chain = LLMChain(llm=llm, prompt=ANSWER_PROMPT)
        logger.info("LLM chain for answer generation configured.")

        # 3. SQL Correction Chain
        CORRECTION_PROMPT_TEXT = """You are an expert PostgreSQL debugger. The user's query failed. Fix the SQL query based on the error.
Rules: Output only the corrected SQL query. Do not add any other text.
PostgreSQL Schema: {schema}
Original Question: {question}
Faulty SQL: {sql_query}
Database Error: {error_message}
Corrected SQL Query:"""
        CORRECTION_PROMPT = PromptTemplate(
            template=CORRECTION_PROMPT_TEXT,
            input_variables=["schema", "question", "sql_query", "error_message"]
        )
        correction_chain = LLMChain(llm=llm, prompt=CORRECTION_PROMPT)
        logger.info("LLM chain for SQL correction configured.")

        # 4. Intelligent Plot Selection Chain
        PLOT_PROMPT_TEXT = """You are a data visualization expert. Your task is to determine the best plot to create based on a user's question and the available data. Your output MUST be a single, clean JSON object.

**Instructions:**
1.  **Analyze Intent:** Does the question ask for a visual ("plot", "chart", "graph", "distribution")? A plot is NOT needed for a single number (e.g., "what is the max/min/avg").
2.  **Choose Plot Type:**
    - `histogram`: For the distribution of one variable.
    - `bar_chart`: For comparing categories.
    - `scatter_geo`: For plotting points on a map.
    - `timeseries`: For a variable over time.
    - `scatter`: For the relationship between two general variables.
    - `profile`: **SPECIAL CASE**: Use ONLY for plotting temperature or salinity vs. depth/pressure.
    - `none`: If no plot is appropriate.
3.  **Select Columns:** Choose the correct column names from "Available Data Columns" for the axes. For `profile` plots, the y-axis MUST be the pressure/depth column (e.g., `pres_dbar`).
4.  **Create Title:** Write a clear, descriptive title.

**Example 1 (Histogram):**
User Question: "Show me the distribution of salinity"
Available Data Columns: ["psal_psu"]
Output:
{{"plot_type": "histogram", "title": "Distribution of Salinity", "x": "psal_psu"}}

**Example 2 (No Plot):**
User Question: "What is the maximum pressure?"
Available Data Columns: ["max_pres_dbar"]
Output:
{{"plot_type": "none"}}

**Example 3 (Profile Plot):**
User Question: "plot temp vs depth graph"
Available Data Columns: ["temp_degc", "pres_dbar"]
Output:
{{"plot_type": "profile", "title": "Temperature vs. Depth Profile", "x": "temp_degc", "y": "pres_dbar"}}

---
**Your Turn:**

User Question: {question}
Available Data Columns: {columns}

Output:
"""
        PLOT_PROMPT = PromptTemplate(template=PLOT_PROMPT_TEXT, input_variables=["question", "columns"])
        plot_selection_chain = LLMChain(llm=intent_llm, prompt=PLOT_PROMPT)
        logger.info("LLM chain for Plot Selection configured.")

    except Exception as e:
        logger.exception("An LLM chain setup failed: %s", e)


def build_context_from_docs(docs: List[Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Builds a structured context string and a provenance list from retrieved documents."""
    if not docs:
        return '', []

    context_parts = [
        f"--- Knowledge from {doc.metadata.get('source', 'unknown')} ---\n{doc.page_content}"
        for doc in docs
    ]
    provenance = [{'source': doc.metadata.get('source'), 'content': doc.page_content} for doc in docs]

    return "\n\n".join(context_parts), provenance
