# services/api/main.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sqlalchemy import text, exc

from .rag import (
    config,
    db_utils,
    llm_utils,
    sql_generator,
    sql_processor,
    response_generator,
    followup_handler
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="FloatChat API")

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

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    question = (req.question or "").strip()
    conv_id = req.conversation_id or str(uuid4())
    diagnostics: List[str] = []
    provenance: List[Dict] = []

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")
    if not db_utils.engine:
        raise HTTPException(status_code=503, detail="Database engine not configured.")

    try:
        # 0. Classify Intent
        intent = "data_query"
        if llm_utils.intent_chain:
            intent = llm_utils.intent_chain.run({"question": question}).strip().lower()
        diagnostics.append(f"Classified intent: {intent}")

        if intent == "greeting":
            return QueryResponse(natural_language_response="Hello! How can I help you with ARGO data today?", conversation_id=conv_id)
        if intent != "data_query":
            return QueryResponse(natural_language_response="I specialize in ARGO data. Please ask me about temperature, salinity, or float locations.", conversation_id=conv_id)

        # 1. Handle Follow-up or Generate New SQL
        sql = ""
        followup_result = followup_handler.handle_followup_request(conv_id, question)
        if followup_result:
            sql = followup_result['sql']
            diagnostics.append("Generated from follow-up conversation.")
        else:
            context = ""
            if llm_utils.retriever:
                docs = llm_utils.retriever.get_relevant_documents(question)
                context, provenance = llm_utils.build_context_from_docs(docs)
                if provenance:
                    diagnostics.append(f"Retrieved {len(provenance)} knowledge documents.")
            
            sql = sql_generator.try_generate_sql_with_llm(context, question)
            if not sql:
                diagnostics.append("LLM SQL generation failed. Falling back to rule-based.")
                sql, diag = sql_generator.rule_based_translator(question, db_utils.introspect_schema())
                diagnostics.extend(diag)

        if not sql:
            return QueryResponse(natural_language_response="I could not translate your question to a query. Please try rephrasing.", conversation_id=conv_id, error="sql_generation_failed")

        # 2. Process, Refine, and Validate SQL
        processed_sql, process_diagnostics = sql_processor.process_and_refine_sql(sql)
        diagnostics.extend(process_diagnostics)

        # 3. Execute SQL with Self-Correction
        df, current_sql = None, processed_sql
        for attempt in range(2):
            try:
                with db_utils.engine.connect() as conn:
                    df = pd.read_sql(text(current_sql), conn)
                diagnostics.append(f"SQL executed successfully on attempt {attempt + 1}.")
                break
            except exc.SQLAlchemyError as e:
                error_str = str(e.orig) if hasattr(e, 'orig') else str(e)
                diagnostics.append(f"SQL attempt {attempt + 1} failed: {error_str}")
                if attempt < 1 and llm_utils.correction_chain:
                    schema_str = "\n".join(db_utils.get_schema_string().values())
                    corrected_sql = llm_utils.correction_chain.run({"schema": schema_str, "question": question, "sql_query": current_sql, "error_message": error_str})
                    current_sql, correction_diags = sql_processor.process_and_refine_sql(corrected_sql)
                    diagnostics.extend(correction_diags)
                    diagnostics.append("SQL was self-corrected.")
                else:
                    raise e

        if df is None: # Should only happen if loop breaks but df is not assigned
             return QueryResponse(natural_language_response="Query executed but failed to produce a result.", sql_query=current_sql, conversation_id=conv_id)

        # 4. Generate Final Response (NL + Visualizations) using the Unified Engine
        nl_response, plots = response_generator.generate_final_response(question, df, current_sql, provenance, diagnostics)
        
        follow_options = followup_handler.build_followup_options(current_sql, question, df)
        followup_handler.store_followups(conv_id, current_sql, follow_options)

        return QueryResponse(
            natural_language_response=nl_response,
            sql_query=current_sql,
            data=response_generator.df_to_json_rows(df.head(100)),
            plots=plots,
            diagnostics=diagnostics,
            provenance=provenance,
            follow_ups=follow_options,
            conversation_id=conv_id
        )

    except Exception as e:
        logger.exception("An unhandled error occurred in /query endpoint: %s", e)
        return QueryResponse(
            natural_language_response=f"I encountered an unexpected server error. Please try your question again.",
            conversation_id=conv_id,
            error=str(e)
        )

@app.get("/health")
def health():
    try:
        with db_utils.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": f"DB connection failed: {e}"}