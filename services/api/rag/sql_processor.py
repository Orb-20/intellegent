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
        
        if base.upper() in config.SQL_KEYWORDS_AND_FUNCTIONS or base in config.SAFE_IDENTIFIERS or base in db_utils.introspect_schema():
            continue
        
        canon = config.SYNONYMS.get(base.lower(), base)
        if canon != base:
            mapping[ident] = canon

        if canon in config.CANONICAL_COLUMNS and '.' not in ident:
            table = config.CANONICAL_COLUMNS[canon]
            qualified_sql = re.sub(rf"\b{re.escape(ident)}\b", f"{table}.{canon}", qualified_sql)
        elif canon not in config.CANONICAL_COLUMNS:
            if not canon.lower() in config.MONTHS and not canon.replace('.', '', 1).isdigit():
                unknown.add(ident)

    return qualified_sql, mapping, sorted(list(unknown))

# --- THIS IS THE CORRECTED FUNCTION ---
def inject_join_if_needed(sql: str) -> Tuple[str, bool]:
    """
    Adds a JOIN to the 'levels' table if its columns are mentioned but the
    table itself is not present in the FROM or JOIN clauses.
    """
    sql_lower = sql.lower()

    # 1. Check if any column requires the 'levels' table.
    # These column names are unique enough to the 'levels' table.
    needs_levels_table = any(col in sql_lower for col in ['temp_degc', 'psal_psu', 'pres_dbar'])

    if not needs_levels_table:
        return sql, False # No join needed, exit early.

    # 2. Check if the 'levels' table is already included in the query.
    # This regex looks for 'FROM levels' or 'JOIN levels' as whole words.
    if re.search(r"\b(from|join)\s+levels\b", sql_lower):
        return sql, False # Table already present, no join needed.

    # 3. If a join is needed and not present, inject it after 'FROM profiles'.
    # This is the safest assumption for where to add the join.
    if "from profiles" in sql_lower:
        # Robustly find 'FROM profiles' and add the JOIN right after it.
        # This handles cases where 'profiles' might have an alias (e.g., "FROM profiles p").
        sql_with_join = re.sub(
            r"(?i)\bfrom\s+profiles\b(?!\.)",
            "FROM profiles JOIN levels ON profiles.profile_id = levels.profile_id",
            sql,
            count=1
        )
        return sql_with_join, True

    # 4. If we can't find 'FROM profiles', don't risk an unsafe modification.
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
                insertion_point = re.search(r"\b(GROUP BY|ORDER BY|LIMIT)\b", q, re.IGNORECASE)
                if insertion_point:
                    idx = insertion_point.start()
                    q = q[:idx] + f" WHERE {filter_cond} " + q[idx:]
                else:
                    q += f" WHERE {filter_cond}"
    return q