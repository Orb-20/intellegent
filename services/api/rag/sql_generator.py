# services/api/rag/sql_generator.py
import logging
from typing import Optional

from . import config
from . import llm_utils

logger = logging.getLogger(__name__)

def try_generate_sql_with_llm(context: str, question: str) -> Optional[str]:
    """
    Attempts to generate SQL using a multi-layered approach for robustness.
    It first tries an advanced prompt and falls back to a simpler one.
    """
    
    # --- Primary Attempt: Use the advanced "expert" prompt ---
    logger.info("Attempting SQL generation with the expert prompt...")
    raw_sql = llm_utils.run_sql_generation(context=context or "", question=question)

    if raw_sql:
        sanitized_sql = config.sanitize_sql(raw_sql)
        if config.is_safe_select(sanitized_sql):
            logger.info("Successfully generated SQL with the expert prompt.")
            return sanitized_sql

    # --- Fallback Attempt: Use the simple prompt safety net ---
    logger.warning("Expert prompt failed. Falling back to the simple prompt safety net.")
    raw_sql_simple = llm_utils.run_simple_sql_generation(question=question)

    if raw_sql_simple:
        sanitized_sql_simple = config.sanitize_sql(raw_sql_simple)
        if config.is_safe_select(sanitized_sql_simple):
            logger.info("Successfully generated SQL with the simple prompt fallback.")
            return sanitized_sql_simple

    logger.error("All SQL generation attempts failed for the question: '%s'", question)
    return None