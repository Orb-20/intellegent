# services/api/main.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import text

# Import modularized components
from .rag import (
    config,
    db_utils,
    llm_utils,
    sql_generator,
    sql_processor,
    response_generator,
    followup_handler
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FloatChat API")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    natural_language_response: str
    sql_query: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    plots: Optional[List[Dict[str, Any]]] = None
    diagnostics: Optional[List[str]] = None
    provenance: Optional[List[Dict[str, Any]]] = None
    follow_ups: Optional[List[Dict[str, Any]]] = None
    conversation_id: str
    error: Optional[str] = None

# --- Main Endpoint ---
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    question = (req.question or "").strip()
    conv_id = req.conversation_id or str(uuid4())
    diagnostics: List[str] = []

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    if not db_utils.engine:
        raise HTTPException(status_code=503, detail="Database engine not configured.")

    # 1. Handle potential follow-up
    followup_result = followup_handler.handle_followup_request(conv_id, question)
    if followup_result:
        sql = followup_result['sql']
        diagnostics.append("Generated from follow-up conversation.")
    else:
        # 2. Generate new SQL if not a follow-up
        context, provenance = "", []
        if llm_utils.retriever:
            try:
                docs = llm_utils.retriever.get_relevant_documents(question)
                context, provenance = llm_utils.build_context_from_docs(docs)
            except Exception as e:
                logger.warning("Retriever error: %s", e)

        sql = sql_generator.try_generate_sql_with_llm(context, question)
        if not sql:
            sql, diag = sql_generator.rule_based_translator(question, db_utils.introspect_schema())
            diagnostics.extend(diag)

    if not sql:
        return QueryResponse(natural_language_response="Could not generate SQL for your question.", conversation_id=conv_id, error="sql_generation_failed")

    # 3. Process and refine SQL
    sql_q, mapping, unknown = sql_processor.qualify_and_validate(sql)
    if mapping: diagnostics.append(f"Applied mappings: {mapping}")
    if unknown: diagnostics.append(f"Unrecognized identifiers: {unknown}")

    sql_q, joined = sql_processor.inject_join_if_needed(sql_q)
    if joined: diagnostics.append("Auto-joined to 'levels' table.")

    sql_q, clamped_diag = sql_processor.clamp_date_range(sql_q)
    if clamped_diag: diagnostics.append(clamped_diag)

    filtered_sql = sql_processor.inject_missing_filters(sql_q)
    if not config.is_safe_select(filtered_sql):
        return QueryResponse(natural_language_response="Generated SQL failed safety checks.", sql_query=filtered_sql, conversation_id=conv_id, error="safety_failed")

    # 4. Execute SQL
    try:
        with db_utils.engine.connect() as conn:
            df = pd.read_sql(text(filtered_sql), conn)
    except Exception as e:
        logger.exception("SQL execution failed: %s", e)
        return QueryResponse(natural_language_response=f"Error executing query: {e}", sql_query=filtered_sql, conversation_id=conv_id, error=str(e), diagnostics=diagnostics)

    # 5. Generate Response
    stats, plots = response_generator.compute_stats_and_plots(df)
    final_plots = response_generator.deduplicate_plots(plots)

    nl_response = followup_result.get('nl_response') if followup_result else response_generator.generate_polished_answer(question, df, filtered_sql, [], diagnostics)
    
    follow_options = followup_handler.build_followup_options(filtered_sql, question, df)
    followup_handler.store_followups(conv_id, filtered_sql, follow_options)

    return QueryResponse(
        natural_language_response=nl_response,
        sql_query=filtered_sql,
        data=response_generator.df_to_json_rows(df),
        plots=final_plots,
        diagnostics=diagnostics,
        follow_ups=follow_options,
        conversation_id=conv_id
    )

@app.get("/health")
def health():
    if not db_utils.engine:
        return {"status": "error", "message": "Database not configured."}
    try:
        with db_utils.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": f"DB connection failed: {e}"}