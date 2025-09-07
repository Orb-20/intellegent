# services/api/rag/sql_processor.py
import re
from typing import List, Tuple, Dict, Set

from . import config
from . import db_utils

# --- Individual Processing Steps (Internal Functions) ---

def _qualify_and_validate(sql: str) -> Tuple[str, Dict[str, str], List[str]]:
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


def _inject_join_if_needed(sql: str) -> Tuple[str, bool]:
    """
    Adds a JOIN between profiles and levels if columns from both are mentioned
    but a join is not already present.
    """
    sql_lower = sql.lower()

    # 1. Check if any column requires the 'levels' table.
    needs_levels_table = any(col in sql_lower for col in ['temp_degc', 'psal_psu', 'pres_dbar'])
    # 2. Check if any column requires the 'profiles' table.
    needs_profiles_table = any(col in sql_lower for col in ['latitude', 'longitude', 'juld', 'platform_number'])

    if not (needs_levels_table and needs_profiles_table):
        return sql, False # No join needed if only one table's columns are used.

    # 3. Check if a join is already present.
    if "join" in sql_lower and "on" in sql_lower:
        return sql, False

    # 4. If a join is needed and not present, inject it.
    if "from profiles" in sql_lower:
        sql_with_join = re.sub(
            r"(?i)\bfrom\s+profiles\b(?!\.)",
            "FROM profiles JOIN levels ON profiles.profile_id = levels.profile_id",
            sql,
            count=1
        )
        return sql_with_join, True

    if "from levels" in sql_lower:
        sql_with_join = re.sub(
            r"(?i)\bfrom\s+levels\b(?!\.)",
            "FROM levels JOIN profiles ON levels.profile_id = profiles.profile_id",
            sql,
            count=1
        )
        return sql_with_join, True

    return sql, False


def _clamp_date_range(sql: str) -> Tuple[str, str]:
    """Clamps the date range in the query to the available data range in the DB."""
    min_juld, max_juld = db_utils.get_juld_range()
    if not (min_juld and max_juld):
        return sql, ""
    
    # Regex to find a complete BETWEEN or >= AND < clause for juld
    pattern = r"profiles\.juld\s+(?:BETWEEN\s+'([^']+)'\s+AND\s+'([^']+)'|>=?\s*'([^']+)'\s+AND\s+profiles\.juld\s+<=?\s*'([^']+)')"
    m = re.search(pattern, sql, re.IGNORECASE)
    if not m:
        return sql, ""

    # Extract start and end dates from whichever group matched
    start = m.group(1) or m.group(3)
    end = m.group(2) or m.group(4)
    
    if not (start and end):
        return sql, ""

    clamped_start, clamped_end = max(start, min_juld), min(end, max_juld)
    
    if clamped_start > clamped_end: # User requested a range entirely outside available data
        clamped_start = clamped_end
        
    if clamped_start != start or clamped_end != end:
        new_clause = f"profiles.juld BETWEEN '{clamped_start}' AND '{clamped_end}'"
        updated_sql = re.sub(pattern, new_clause, sql, flags=re.IGNORECASE, count=1)
        diagnostic = f"Clamped date range to available data ({min_juld[:10]} to {max_juld[:10]})."
        return updated_sql, diagnostic
        
    return sql, ""

def _inject_missing_filters(sql: str) -> str:
    """Adds WHERE conditions to filter out missing/sentinel values."""
    q = sql
    measurement_cols = ["levels.temp_degc", "levels.psal_psu", "levels.pres_dbar"]
    for col in measurement_cols:
        # Check if the column is mentioned and not already filtered
        if col in q.lower() and not re.search(rf"\b{re.escape(col)}\b\s*[<>=]", q, re.IGNORECASE):
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

# --- UNIFIED PIPELINE FUNCTION ---
def process_and_refine_sql(sql: str) -> Tuple[str, List[str]]:
    """
    Runs a generated SQL query through a complete, standardized pipeline of
    validation, refinement, and filtering steps. This ensures consistency.
    Returns the final, processed SQL and a list of diagnostics.
    """
    diagnostics = []
    
    # Clean up artifacts from LLM generation (e.g., ```sql markers)
    cleaned_sql = re.sub(r"```sql\n|```", "", sql).strip()
    
    # Stage 1: Qualify column names and validate identifiers
    sql_q, mapping, unknown = _qualify_and_validate(cleaned_sql)
    if mapping: diagnostics.append(f"Applied mappings: {mapping}")
    if unknown: diagnostics.append(f"Unrecognized identifiers: {unknown}")

    # Stage 2: Ensure necessary joins are present
    sql_q, joined = _inject_join_if_needed(sql_q)
    if joined: diagnostics.append("Auto-joined 'profiles' and 'levels' tables.")

    # Stage 3: Clamp date ranges to available data
    sql_q, clamped_diag = _clamp_date_range(sql_q)
    if clamped_diag: diagnostics.append(clamped_diag)

    # Stage 4: Inject critical filters for data quality (e.g., remove sentinels)
    final_sql = _inject_missing_filters(sql_q)
    if final_sql != sql_q:
        diagnostics.append("Applied mandatory data quality filters.")

    # Stage 5: Final safety check
    if not config.is_safe_select(final_sql):
        raise ValueError("Processed SQL failed safety check.")

    return final_sql, diagnostics