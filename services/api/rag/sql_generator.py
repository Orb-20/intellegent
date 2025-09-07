# services/api/rag/sql_generator.py
import re
import datetime as dt
import logging
from typing import List, Tuple, Dict, Optional

from . import config
from . import llm_utils

logger = logging.getLogger(__name__)

def try_generate_sql_with_llm(context: str, question: str) -> str:
    """Attempts to generate SQL using the configured LLM chain."""
    if not llm_utils.llm_chain:
        return ""
    try:
        raw_sql = llm_utils.llm_chain.run({"context": context or "", "question": question})
        sanitized = config.sanitize_sql(raw_sql)
        if config.is_safe_select(sanitized):
            return sanitized
        logger.warning("LLM produced unsafe or invalid SQL: %s", raw_sql)
    except Exception as e:
        logger.exception("LLM SQL generation failed: %s", e)
    return ""

def _detect_date_range(text: str) -> Optional[Tuple[str, str]]:
    """Detects a date range from text (e.g., 'in March 2022')."""
    text_l = text.lower()
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{4})", text_l)
    if m:
        month_str, year_str = m.group(1), m.group(2)
        month = next((v for k, v in config.MONTHS.items() if k.startswith(month_str[:3])), None)
        year = int(year_str)
        if month:
            start = dt.date(year, month, 1)
            end_month, end_year = (1, year + 1) if month == 12 else (month + 1, year)
            end = dt.date(end_year, end_month, 1)
            return start.isoformat(), end.isoformat()
    return None

def rule_based_translator(question: str, schema: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    """A fallback rule-based system to translate natural language to SQL."""
    q_lower = question.lower()
    diagnostics: List[str] = []

    # Detect aggregate intent
    agg = None
    if any(k in q_lower for k in ["max","maximum","highest"]): agg = 'MAX'
    elif any(k in q_lower for k in ["min","minimum","lowest"]): agg = 'MIN'
    elif any(k in q_lower for k in ["avg","average","mean"]): agg = 'AVG'
    elif any(k in q_lower for k in ["count","how many"]): agg = 'COUNT'

    # Identify requested columns
    detected_cols = {canon for token, canon in config.SYNONYMS.items() if f" {token} " in f" {q_lower} "}
    if not detected_cols:
        detected_cols = {'profile_id', 'juld', 'latitude', 'longitude'}
        diagnostics.append('No variable detected; defaulting to profile summary.')

    # Build SELECT clause
    select_parts = []
    if agg:
        if agg == 'COUNT':
            select_parts.append("COUNT(DISTINCT profiles.profile_id) as count_profiles")
        else:
            for col in detected_cols:
                if config.CANONICAL_COLUMNS.get(col) == 'levels':
                    select_parts.append(f"{agg}(levels.{col}) as {agg.lower()}_{col}")
        if not select_parts:
             select_parts.append(f"{agg}(*) as result")
    else:
        for col in detected_cols:
            table = config.CANONICAL_COLUMNS.get(col, "profiles")
            select_parts.append(f"{table}.{col}")

    select_clause = ', '.join(dict.fromkeys(select_parts)) # Remove duplicates

    # Build FROM and JOIN
    needs_levels = any(config.CANONICAL_COLUMNS.get(c) == 'levels' for c in detected_cols)
    from_clause = 'profiles'
    join_clause = ' JOIN levels ON profiles.profile_id = levels.profile_id' if needs_levels else ''

    # Build WHERE
    where_clauses = []
    date_range = _detect_date_range(q_lower)
    if date_range:
        where_clauses.append(f"profiles.juld >= '{date_range[0]}' AND profiles.juld < '{date_range[1]}'")
    where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # Assemble query
    limit_clause = f" LIMIT {config.DEFAULT_LIMIT}" if not agg else ""
    sql = f"SELECT {select_clause} FROM {from_clause}{join_clause}{where_str}{limit_clause}"

    return config.sanitize_sql(sql), diagnostics