# services/api/rag/sql_processor.py
import re
from typing import List, Tuple, Dict

from . import config
from . import db_utils

def _remove_sql_comments_and_markdown(sql: str) -> str:
    """
    Cleans the raw LLM output by removing markdown fences and SQL comments.
    This is the crucial first step to prevent these artifacts from interfering
    with subsequent processing.
    """
    # Remove markdown code fences (e.g., ```sql ... ```)
    sql = re.sub(r"```sql\n|```", "", sql).strip()
    # Remove single-line comments (e.g., -- get all profiles)
    sql = re.sub(r"--.*", "", sql)
    # Remove multi-line comments (e.g., /* comment block */)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip()

def _inject_join_if_needed(sql: str) -> Tuple[str, bool]:
    """
    Automatically adds a JOIN between the 'profiles' and 'levels' tables if
    columns from both are mentioned but a JOIN clause is missing.
    """
    sql_lower = sql.lower()
    
    # Check if columns from both primary tables are present
    needs_levels_table = any(col in sql_lower for col in ['temp_degc', 'psal_psu', 'pres_dbar'])
    needs_profiles_table = any(col in sql_lower for col in ['latitude', 'longitude', 'juld', 'platform_number'])

    # If a join isn't needed or is already present, do nothing
    if not (needs_levels_table and needs_profiles_table) or "join" in sql_lower:
        return sql, False

    # Intelligently inject the join clause after the 'profiles' table reference
    if "from profiles" in sql_lower:
        sql_with_join = re.sub(
            r"(?i)\bfrom\s+profiles\b", 
            "FROM profiles JOIN levels ON profiles.profile_id = levels.profile_id", 
            sql, 
            count=1
        )
        return sql_with_join, True

    return sql, False

def _inject_missing_filters(sql: str) -> str:
    """
    Injects mandatory data quality filters (e.g., for sentinel values)
    into the SQL query in a robust and safe manner.
    """
    # Define which columns need a mandatory quality filter
    measurement_cols = ["levels.temp_degc", "levels.psal_psu", "levels.pres_dbar"]
    
    # Build a list of filters that are needed based on the query content
    required_filters = []
    for col in measurement_cols:
        col_name = col.split('.')[-1]
        # Check if the column is in the query and doesn't already have a filter
        if col_name in sql.lower() and not re.search(rf"\b{col_name}\b\s*[<>=!]", sql, re.IGNORECASE):
            required_filters.append(f"{col} < {config.MISSING_THRESHOLD}")
    
    if not required_filters:
        return sql

    filter_clause = " AND ".join(required_filters)
    sql_lower = sql.lower()

    # If a WHERE clause already exists, append our filters with AND
    if " where " in sql_lower:
        return re.sub(r"(?i)\bwhere\b", f"WHERE {filter_clause} AND ", sql, 1)
    
    # If no WHERE clause, we need to insert one before other clauses
    else:
        # Find the first occurrence of a clause that must come after WHERE
        clause_pattern = r"\b(group by|order by|limit|having)\b"
        match = re.search(clause_pattern, sql_lower)
        
        if match:
            # Insert the WHERE clause before the found clause
            insertion_point = match.start()
            return f"{sql[:insertion_point]}WHERE {filter_clause} {sql[insertion_point:]}"
        else:
            # If no other clauses, simply append the WHERE clause
            return f"{sql} WHERE {filter_clause}"

def process_and_refine_sql(sql: str) -> Tuple[str, List[str]]:
    """
    Runs a generated SQL query through a complete, standardized pipeline of cleaning,
    refinement, and validation to ensure it is safe and correct before execution.
    
    This is the main orchestration function for this module.
    """
    diagnostics = []
    
    # STAGE 1: Initial Cleaning. This is the most important first step.
    if not sql:
        raise ValueError("Received an empty SQL query.")
    cleaned_sql = _remove_sql_comments_and_markdown(sql)
    if not cleaned_sql:
        raise ValueError("SQL query is empty after cleaning comments and markdown.")

    # STAGE 2: Auto-Join tables if necessary.
    sql_with_joins, did_join = _inject_join_if_needed(cleaned_sql)
    if did_join:
        diagnostics.append("Automatically joined 'profiles' and 'levels' tables.")

    # STAGE 3: Inject Mandatory Data Quality Filters. This ensures scientific correctness.
    sql_with_filters = _inject_missing_filters(sql_with_joins)
    if sql_with_filters != sql_with_joins:
        diagnostics.append("Applied mandatory data quality filters for sentinel values.")

    # STAGE 4: Final Validation and Safety Check.
    final_sql = sql_with_filters
    if not config.is_safe_select(final_sql):
        # Provide a highly descriptive error for easier debugging.
        error_message = f"Processed SQL failed the final safety check. This can happen with requests that are not simple SELECT queries. Rejected SQL: '{final_sql}'"
        raise ValueError(error_message)

    diagnostics.append("SQL query passed all processing and safety checks.")
    return final_sql, diagnostics