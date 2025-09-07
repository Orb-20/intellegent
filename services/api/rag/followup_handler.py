# services/api/rag/followup_handler.py
import time
import re
from typing import List, Dict, Any, Optional
from uuid import uuid4

# In a production environment, replace this with a Redis or other persistent store
FOLLOWUP_STORE: Dict[str, Dict[str, Any]] = {}

from . import config

def _prune_followups():
    """Removes expired entries from the in-memory follow-up store."""
    now = time.time()
    expired = [cid for cid, v in FOLLOWUP_STORE.items() if now - v.get("ts", 0) > config.FOLLOWUP_TTL_SECONDS]
    for cid in expired:
        FOLLOWUP_STORE.pop(cid, None)

def _extract_where_clause(sql: str) -> str:
    """Extracts the WHERE clause from a SQL query."""
    mwhere = re.search(r"(WHERE\s+.+?)(?:\bGROUP BY\b|\bORDER BY\b|\bLIMIT\b|$)", sql, flags=re.I | re.S)
    return mwhere.group(1) if mwhere else ""

def store_followups(conversation_id: str, filtered_sql: str, options: List[Dict[str, str]]):
    """Stores the context for a follow-up conversation."""
    _prune_followups()
    where_clause = _extract_where_clause(filtered_sql)
    FOLLOWUP_STORE[conversation_id] = {
        "ts": time.time(),
        "filtered_sql_where": where_clause,
        "options": options
    }

def build_followup_options(sql: str, question: str, df) -> List[Dict[str, str]]:
    """Generates a list of potential follow-up questions."""
    options: List[Dict[str, str]] = []
    q_lower = question.lower()
    
    is_agg = any(agg in q_lower for agg in ("average", "maximum", "minimum", "count"))
    if is_agg:
        options.append({"id": "profile_count", "text": "How many profiles were used?"})
        options.append({"id": "depth_breakdown", "text": "Show a depth-binned breakdown."})

    if df is not None and 'juld' in df.columns:
        options.append({"id": "timeseries_plot", "text": "Plot this as a timeseries."})
        
    return options

def handle_followup_request(conversation_id: str, user_text: str) -> Optional[Dict[str, Any]]:
    """
    Handles a follow-up request by generating a new SQL query based on stored context.
    Returns a dictionary with 'sql' and 'response' keys if a follow-up is handled.
    """
    _prune_followups()
    entry = FOLLOWUP_STORE.get(conversation_id)
    if not entry or not user_text:
        return None

    # Simple matching logic
    text_lower = user_text.lower()
    chosen_id = None
    for option in entry.get("options", []):
        if option['id'] in text_lower or option['text'].lower() in text_lower:
            chosen_id = option['id']
            break

    if not chosen_id:
        return None

    where_clause = entry.get("filtered_sql_where", "")
    sql, response = None, None

    if chosen_id == "profile_count":
        sql = f"SELECT COUNT(DISTINCT profiles.profile_id) as result FROM profiles JOIN levels ON profiles.profile_id = levels.profile_id {where_clause}"
        response = "Counting the number of unique profiles for your previous query."
    elif chosen_id == "depth_breakdown":
        sql = (f"SELECT (floor(levels.pres_dbar/100)*100) AS depth_bin, "
               f"AVG(levels.temp_degc) AS avg_temp, AVG(levels.psal_psu) AS avg_psal, COUNT(*) AS n_points "
               f"FROM profiles JOIN levels ON profiles.profile_id = levels.profile_id {where_clause} "
               f"GROUP BY depth_bin ORDER BY depth_bin")
        response = "Generating a depth-binned breakdown (100 dbar bins)."

    if sql and response:
        return {"sql": config.sanitize_sql(sql), "nl_response": response}

    return None