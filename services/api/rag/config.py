# services/api/rag/config.py
import os
import re
from typing import Dict, Set

# --- Environment Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DB_USER = os.getenv("POSTGRES_USER", "agro")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_NAME = os.getenv("POSTGRES_DB", "agro")
DB_HOST = os.getenv("POSTGRES_HOST", "db")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "floatchat")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "200"))
MISSING_THRESHOLD = 9000
FILL_VALUE = int(os.getenv("FILL_VALUE", "99999"))
FOLLOWUP_TTL_SECONDS = 10 * 60  # 10 minutes

# --- Schema & Synonyms ---
CANONICAL_COLUMNS: Dict[str, str] = {
    # Profiles table
    "profile_id": "profiles", "source_file": "profiles", "profile_index_in_file": "profiles",
    "platform_number": "profiles", "cycle_number": "profiles", "juld": "profiles",
    "latitude": "profiles", "longitude": "profiles", "direction": "profiles",
    "data_mode": "profiles", "project_name": "profiles", "geom": "profiles",
    # Levels table
    "level_id": "levels", "level_index": "levels", "pres_dbar": "levels",
    "temp_degc": "levels", "psal_psu": "levels"
}

SYNONYMS: Dict[str, str] = {
    "salinity": "psal_psu", "sal": "psal_psu", "psu": "psal_psu",
    "temperature": "temp_degc", "temp": "temp_degc",
    "pressure": "pres_dbar", "pres": "pres_dbar", "depth": "pres_dbar",
    "time": "juld", "date": "juld", "year": "juld", "month": "juld",
    "lat": "latitude", "lon": "longitude", "long": "longitude", "location": "geom"
}

# --- Validation & Safety ---
SAFE_IDENTIFIERS: Set[str] = {
    "argo", "float", "floats", "dataset", "data", "profile", "profiles", "levels", "result"
}

SQL_KEYWORDS_AND_FUNCTIONS: Set[str] = {k.upper() for k in [
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET",
    "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "ON",
    "AS", "AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "IS", "NULL",
    "CASE", "WHEN", "THEN", "ELSE", "END",
    "WITH", "DISTINCT", "ASC", "DESC", "COUNT", "SUM", "AVG", "MIN", "MAX",
    "EXTRACT", "CAST"
]}

# --- Utilities ---
MONTHS: Dict[str, int] = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

def sanitize_sql(s: str) -> str:
    """Cleans up SQL string by removing extra whitespace and trailing semicolons."""
    return re.sub(r"\s+", " ", s.strip().rstrip(';')).strip()

def is_safe_select(sql: str) -> bool:
    """Performs a basic safety check on the generated SQL."""
    if not sql:
        return False
    su = sql.upper()
    if not (su.startswith("SELECT") or su.startswith("WITH")):
        return False
    # Check for forbidden keywords and multiple statements
    if any(k in su for k in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", ";", "--", "/*"]):
        return False
    return len(sql) < 20000