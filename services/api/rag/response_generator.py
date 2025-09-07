import json
import logging
# --- THIS LINE IS CORRECTED ---
from typing import Any, List, Dict, Optional, Tuple

import pandas as pd
from . import config, llm_utils

logger = logging.getLogger(__name__)

def df_to_json_rows(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Converts a DataFrame to a list of JSON-serializable dictionaries."""
    if df is None or df.empty:
        return []
    
    def normalize_value(v):
        if pd.isna(v): return None
        if hasattr(v, 'item'): return v.item()
        return v

    return [{c: normalize_value(row[c]) for c in df.columns} for _, row in df.iterrows()]

def compute_stats_and_plots(df: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Generates summary statistics and basic plot specifications from a DataFrame."""
    stats, plots = {}, []
    if df is None or df.empty:
        return stats, plots

    try:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            stats = df[numeric_cols].describe().round(3).to_dict()

        # Timeseries plot for high-variance numeric columns if 'juld' exists
        if 'juld' in df.columns and numeric_cols:
            df['juld'] = pd.to_datetime(df['juld'], errors='coerce')
            var_cols = df[numeric_cols].var().nlargest(2).index.tolist()
            for col in var_cols:
                sub = df.dropna(subset=['juld', col]).sort_values('juld')
                if not sub.empty:
                    plots.append({
                        "type": "timeseries", "title": f"{col} over time",
                        "x": sub['juld'].dt.strftime('%Y-%m-%dT%H:%M:%SZ').tolist(),
                        "y": sub[col].tolist(), "y_label": col
                    })

        # Geo scatter plot if lat/lon exist
        if 'latitude' in df.columns and 'longitude' in df.columns:
            geo = df.dropna(subset=['latitude', 'longitude'])
            if not geo.empty:
                 plots.append({
                    "type": "scattergeo", "title": "Map of points",
                    "lat": geo['latitude'].tolist(), "lon": geo['longitude'].tolist(),
                    "label": geo.get('profile_id', geo.index).astype(str).tolist()
                })
    except Exception as e:
        logger.exception('compute_stats_and_plots error: %s', e)
        
    return stats, plots

def generate_polished_answer(question: str, df: Optional[pd.DataFrame], sql: str, provenance: List[Dict], diagnostics: List[str]) -> str:
    """Generates a polished, natural language summary of the results using an LLM."""
    if df is None or df.empty:
        return "No results found for your query. You could try rephrasing or broadening the criteria."

    if llm_utils.LANGCHAIN_AVAILABLE and config.GOOGLE_API_KEY:
        try:
            stats, _ = compute_stats_and_plots(df)
            data_summary = f"Query returned {len(df)} rows."
            if stats:
                data_summary += f" Key stats: {json.dumps(stats, default=str)}"

            prompt = (
                "You are FloatChat, an expert data analyst for ARGO floats.\n"
                "Instructions:\n"
                "- Produce a concise (1-2 sentence) answer to the user's question based on the data summary.\n"
                "- Do NOT repeat the SQL query or raw diagnostics. Use them only for context.\n"
                "- Suggest a single, relevant follow-up question.\n\n"
                f"User question: {question}\n"
                f"Data summary: {data_summary}\n\n"
                "Answer:"
            )
            # This logic was referencing a non-existent llm_utils.ChatGoogleGenerativeAI, so I've corrected it.
            if llm_utils.llm_chain:
                 return llm_utils.llm_chain.run({"context": data_summary, "question": question})

        except Exception as e:
            logger.exception("LLM polish failed: %s", e)
            
    # Fallback response
    return f"The query returned {len(df)} rows. You can ask to visualize these results or request aggregates (max/min/avg)."

def deduplicate_plots(plots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes duplicate plots from a list."""
    final_plots: List[Dict[str, Any]] = []
    seen = set()
    for p in plots:
        h = json.dumps(p, sort_keys=True, default=str)
        if h not in seen:
            seen.add(h)
            final_plots.append(p)
    return final_plots