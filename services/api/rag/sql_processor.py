# services/api/rag/sql_processor.py
import re
from typing import List, Tuple, Dict, Set

from . import config
from . import db_utils

def qualify_and_validate(sql: str) -> Tuple[str, Dict[str, str], List[str]]:
    """Qualifies column names with table names and validates identifiers."""
    mapping: Dict[str, str] = {}
    unknown: Set[str] = set()
    if not sql:
        return "", mapping, []

    qualified_sql = sql
    identifiers = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_\.]*", sql))

    for ident in sorted(identifiers, key=len, reverse=True):
        base = ident.split('.')[-1]
        
        # Skip SQL keywords, known tables, and safe identifiers
        if base.upper() in config.SQL_KEYWORDS_AND_FUNCTIONS or base in config.SAFE_IDENTIFIERS or base in db_utils.introspect_schema():
            continue
        
        # Apply synonyms
        canon = config.SYNONYMS.get(base.lower(), base)
        if canon != base:
            mapping[ident] = canon

        # Qualify column if not already qualified
        if canon in config.CANONICAL_COLUMNS and '.' not in ident:
            table = config.CANONICAL_COLUMNS[canon]
            qualified_sql = re.sub(rf"\b{re.escape(ident)}\b", f"{table}.{canon}", qualified_sql)
        elif canon not in config.CANONICAL_COLUMNS:
            if not canon.lower() in config.MONTHS and not canon.replace('.', '', 1).isdigit():
                unknown.add(ident)

    return qualified_sql, mapping, sorted(list(unknown))

def inject_join_if_needed(sql: str) -> Tuple[str, bool]:
    """Adds a JOIN to the 'levels' table if measurement variables are used without a join."""
    sql_lower = sql.lower()
    needs_join = any(f"levels.{col}" in sql_lower for col in ["temp_degc", "psal_psu", "pres_dbar", "level_index"])
    
    if needs_join and "join levels" not in sql_lower:
        if re.search(r"(?i)\bFROM\s+profiles\b", sql):
            sql = re.sub(r"(?i)\bFROM\s+profiles\b", "FROM profiles JOIN levels ON profiles.profile_id = levels.profile_id", sql, count=1)
            return sql, True
        else: # Fallback for more complex FROM clauses
            sql = re.sub(r"(?i)\bFROM\s+([a-zA-Z0-9_\.]+)", lambda m: f"FROM {m.group(1)} JOIN levels ON {m.group(1)}.profile_id = levels.profile_id", sql, count=1)
            return sql, True
    return sql, False

def clamp_date_range(sql: str) -> Tuple[str, str]:
    """Clamps the date range in the query to the available data range in the DB."""
    min_juld, max_juld = db_utils.get_juld_range()
    if not (min_juld and max_juld):
        return sql, ""
    
    m = re.search(r"profiles\.juld\s*(>=|BETWEEN)\s*'([^']+)'\s*AND\s*profiles\.juld\s*(<|<=)\s*'([^']+)'", sql, re.IGNORECASE)
    if not m:
        return sql, ""

    start, end = m.group(2), m.group(4)
    clamped_start, clamped_end = max(start, min_juld), min(end, max_juld)
    
    if clamped_start != start or clamped_end != end:
        new_clause = f"profiles.juld >= '{clamped_start}' AND profiles.juld < '{clamped_end}'"
        updated_sql = re.sub(r"profiles\.juld\s*(>=|BETWEEN)\s*'[^']+'\s*AND\s*profiles\.juld\s*(<|<=)\s*'[^']+'", new_clause, sql, flags=re.IGNORECASE)
        diagnostic = f"Clamped date range to available data ({min_juld} to {max_juld})."
        return updated_sql, diagnostic
        
    return sql, ""

def inject_missing_filters(sql: str) -> str:
    """Adds WHERE conditions to filter out missing/sentinel values."""
    q = sql
    measurement_cols = ["levels.temp_degc", "levels.psal_psu", "levels.pres_dbar"]
    for col in measurement_cols:
        if col in q.lower() and not re.search(rf"\b{re.escape(col)}\b\s*<\s*{config.MISSING_THRESHOLD}", q, re.IGNORECASE):
            filter_cond = f"({col} IS NOT NULL AND {col} < {config.MISSING_THRESHOLD})"
            if " where " in q.lower():
                q = re.sub(r"(?i)\bWHERE\b", f"WHERE {filter_cond} AND ", q, count=1)
            else:
                # Find a place to insert WHERE (e.g., before GROUP BY, ORDER BY, LIMIT)
                insertion_point = re.search(r"\b(GROUP BY|ORDER BY|LIMIT)\b", q, re.IGNORECASE)
                if insertion_point:
                    idx = insertion_point.start()
                    q = q[:idx] + f" WHERE {filter_cond} " + q[idx:]
                else:
                    q += f" WHERE {filter_cond}"
    return q