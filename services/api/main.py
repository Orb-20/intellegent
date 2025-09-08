# services/api/main.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # <-- CORRECTED TYPO
import pandas as pd
from sqlalchemy import text, exc

from .rag import (
    config,
    db_utils,
    llm_utils,
    sql_generator,
    sql_processor,
    response_generator
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
    conversation_id: str
    error: Optional[str] = None

def _is_analytical_profile_request(question: str, sql: str) -> bool:
    """
    Expert heuristic to detect if the user is asking for a broad analytical plot
    that requires advanced, database-side statistical analysis.
    """
    q_lower = question.lower()
    sql_lower = sql.lower()
    analytical_keywords = ["relation", "relationship", "trend", "correlation", "vs", "versus", "variation", "distribution"]
    has_profile_columns = 'pres_dbar' in sql_lower and ('temp_degc' in sql_lower or 'psal_psu' in sql_lower)
    is_broad_query = "where" not in sql_lower or "between" in sql_lower or "profile_id =" not in sql_lower
    return any(k in q_lower for k in analytical_keywords) and has_profile_columns and is_broad_query

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    question = (req.question or "").strip()
    conv_id = req.conversation_id or str(uuid4())
    diagnostics: List[str] = []
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        initial_sql = sql_generator.try_generate_sql_with_llm("", question)
        if not initial_sql:
            return QueryResponse(natural_language_response="I could not translate your question into a valid query.", conversation_id=conv_id)

        processed_sql, process_diags = sql_processor.process_and_refine_sql(initial_sql)
        diagnostics.extend(process_diags)

        final_sql_to_execute = processed_sql
        is_analytical_query = _is_analytical_profile_request(question, processed_sql)
        
        if is_analytical_query:
            diagnostics.append("Analytical intent detected. Rewriting SQL for advanced database-side statistical analysis.")
            measure_col_name = 'temp_degc' if 'temp_degc' in processed_sql.lower() else 'psal_psu'
            
            # This is the improved, more efficient analytical query.
            final_sql_to_execute = f"""
            WITH base_data AS (
                {processed_sql.replace(';', '')}
            ),
            binned_data AS (
                SELECT
                    *,
                    floor(pres_dbar / 50) * 50 AS pres_bin -- Binning by 50 dbar for a clear trend
                FROM base_data
                WHERE pres_dbar IS NOT NULL AND {measure_col_name} IS NOT NULL
            )
            SELECT
                pres_bin,
                AVG({measure_col_name}) as mean,
                STDDEV({measure_col_name}) as std,
                MIN({measure_col_name}) as min_val,
                MAX({measure_col_name}) as max_val,
                COUNT(*) as count
            FROM binned_data
            GROUP BY pres_bin
            HAVING COUNT(*) > 100 -- Ensure statistical significance
            ORDER BY pres_bin;
            """
            diagnostics.append(f"Optimized Analytical SQL Executed.")

        with db_utils.engine.connect() as conn:
            df = pd.read_sql(text(final_sql_to_execute), conn)
        
        if is_analytical_query:
            df.attrs['is_analytical'] = True
            df.attrs['x_measure'] = measure_col_name

        nl_response, plots = response_generator.generate_final_response(question, df, processed_sql, [], diagnostics)

        return QueryResponse(
            natural_language_response=nl_response,
            sql_query=processed_sql,
            data=response_generator.df_to_json_rows(df.head(100)),
            plots=plots,
            diagnostics=diagnostics,
            conversation_id=conv_id
        )

    except Exception as e:
        logger.exception("An unhandled error occurred in /query endpoint")
        return QueryResponse(
            natural_language_response=f"Sorry, I encountered a server error: {type(e).__name__}. Please check the logs for details.",
            conversation_id=conv_id,
            error=str(e)
        )