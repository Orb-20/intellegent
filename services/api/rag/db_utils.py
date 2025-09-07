# services/api/rag/db_utils.py
import logging
from typing import Dict, List, Optional, Tuple

from sqlalchemy import create_engine, inspect, text
from . import config

logger = logging.getLogger(__name__)

# --- Database Engine Initialization ---
try:
    engine = create_engine(config.DB_URL)
    logger.info("SQLAlchemy engine configured for database: %s", config.DB_NAME)
except Exception as e:
    logger.exception("Failed to configure SQLAlchemy engine: %s", e)
    engine = None

# --- Schema Helpers ---
def introspect_schema() -> Dict[str, List[str]]:
    """Inspects the database and returns a dictionary of tables and their columns."""
    schema: Dict[str, List[str]] = {}
    if not engine:
        return schema
    try:
        inspector = inspect(engine)
        for t in inspector.get_table_names():
            cols = [c['name'] for c in inspector.get_columns(t)]
            schema[t] = cols
    except Exception as e:
        logger.warning("Schema introspection failed: %s", e)
    return schema

# --- THIS IS THE NEW FUNCTION ---
def get_schema_string() -> Dict[str, str]:
    """
    Returns a dictionary of table names to their CREATE TABLE like string representation.
    This is useful for providing schema context to the LLM.
    """
    schema_strings: Dict[str, str] = {}
    if not engine:
        return schema_strings
    try:
        inspector = inspect(engine)
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            col_defs = [f"{c['name']} {c['type']}" for c in columns]
            schema_strings[table_name] = f"Table {table_name}({', '.join(col_defs)})"
    except Exception as e:
        logger.warning("get_schema_string failed: %s", e)
    return schema_strings


def get_juld_range() -> Tuple[Optional[str], Optional[str]]:
    """Gets the min and max 'juld' timestamps from the profiles table."""
    if not engine:
        return None, None
    try:
        with engine.connect() as conn:
            r = conn.execute(text("SELECT MIN(juld) as minj, MAX(juld) as maxj FROM profiles"))
            row = r.fetchone()
            if row and row[0] is not None:
                return str(row[0]), str(row[1])
    except Exception as e:
        logger.debug("get_juld_range failed: %s", e)
    return None, None