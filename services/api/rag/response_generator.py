# services/api/rag/response_generator.py
import json
import logging
import re
from typing import Any, List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import text

from . import config, llm_utils, db_utils

logger = logging.getLogger(__name__)

MAX_RAW_DATA_POINTS = 500  # Threshold for triggering backend aggregation
PROFILE_PLOT_BINS = 100    # Number of depth bins for aggregated profile plots


def _safe_json_loads(s: str) -> Optional[Dict]:
    """Safely decodes a JSON string that might be embedded in markdown or have surrounding text."""
    try:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        logger.warning(f"Could not find a valid JSON object in the LLM response: {s}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from LLM response: {s}")
        return None


def df_to_json_rows(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    """Converts a DataFrame to a list of JSON-serializable dictionaries."""
    if df is None or df.empty:
        return []

    def normalize_value(v):
        if pd.isna(v):
            return None
        if hasattr(v, "item"):
            return v.item()
        return v

    return [{c: normalize_value(row[c]) for c in df.columns} for _, row in df.iterrows()]


def generate_final_response(
    question: str,
    df: pd.DataFrame,
    sql: str,
    provenance: List[Dict],
    diagnostics: List[str],
) -> Tuple[str, List[Dict]]:
    """
    Unified pipeline to generate the final NL response and visualizations.
    1. Computes stats for context.
    2. Generates a polished NL answer using the stats.
    3. Intelligently selects and generates optimized visualizations.
    """
    # --- Part 1: Generate Natural Language Answer ---
    nl_response = f"The query returned {len(df)} rows."  # Default fallback
    try:
        data_summary = f"Query returned {len(df)} rows."
        if not df.empty:
            if len(df) == 1 and len(df.columns) == 1:
                single_value = df.iloc[0, 0]
                data_summary += f" The result is a single value: {single_value:.2f}"
            else:
                numeric_stats = df.describe().round(2).to_dict()
                if numeric_stats:
                    data_summary += f" Key stats: {json.dumps(numeric_stats, default=str)}"

        if llm_utils.answer_chain:
            nl_response = llm_utils.answer_chain.run(
                {"context": data_summary, "question": question}
            )
        else:
            logger.warning("Answer chain not available, using fallback response.")

    except Exception as e:
        logger.exception("Error during polished answer generation: %s", e)

    # --- Part 2: Generate Visualizations with Intelligent Aggregation ---
    plots = []
    if df.empty or not llm_utils.plot_selection_chain:
        return nl_response, plots

    try:
        cols = df.columns.tolist()
        plot_spec_str = llm_utils.plot_selection_chain.run(
            {"question": question, "columns": str(cols)}
        )
        plot_spec = _safe_json_loads(plot_spec_str)

        if not plot_spec or plot_spec.get("plot_type") == "none":
            return nl_response, []

        logger.info(f"AI selected plot: {plot_spec}")
        plot_spec["data"] = []
        plot_type = plot_spec.get("plot_type")

        # --- INTELLIGENT AGGREGATION STRATEGY ---
        if plot_type == "histogram" and len(df) > MAX_RAW_DATA_POINTS:
            # Optimized backend histogram query
            x_col = plot_spec.get("x")
            if x_col and x_col in df.columns and pd.api.types.is_numeric_dtype(df[x_col]):
                from_where_clause = re.search(
                    r"\bFROM\b.*", sql, re.IGNORECASE | re.DOTALL
                ).group(0)
                min_val, max_val = df[x_col].min(), df[x_col].max()
                if pd.notna(min_val) and pd.notna(max_val) and min_val < max_val:
                    bin_size = (max_val - min_val) / 50
                    if bin_size > 0:
                        agg_sql = f"""
                        SELECT floor({x_col} / {bin_size}) * {bin_size} as bin_start,
                               COUNT(*) as frequency
                        {from_where_clause}
                        GROUP BY bin_start
                        ORDER BY bin_start
                        """
                        logger.info("Running optimized aggregation query for histogram.")
                        with db_utils.engine.connect() as conn:
                            agg_df = pd.read_sql(text(agg_sql), conn)
                        plot_spec["data"] = agg_df.to_dict(orient="records")
                        plot_spec["plot_strategy"] = "aggregated_histogram"
                        plots.append(plot_spec)

        elif plot_type == "profile" and len(df) > MAX_RAW_DATA_POINTS:
            # Summarized average profile (mean + std over bins)
            y_col = plot_spec.get("y")  # Pressure/Depth
            x_col = plot_spec.get("x")  # Temperature/Salinity
            if y_col and x_col and y_col in df.columns and x_col in df.columns:
                logger.info(
                    f"Data too large for profile plot ({len(df)} points). Generating aggregated summary."
                )
                min_pres = df[y_col].min()
                max_pres = df[y_col].max()
                bins = np.linspace(min_pres, max_pres, PROFILE_PLOT_BINS)
                df["pres_bin"] = pd.cut(df[y_col], bins)

                agg_df = (
                    df.groupby("pres_bin", observed=True)[x_col]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                agg_df["pres_bin_mid"] = agg_df["pres_bin"].apply(lambda b: b.mid).astype(float)
                agg_df["std"] = agg_df["std"].fillna(0)

                plot_spec["data"] = agg_df.to_dict(orient="records")
                plot_spec["plot_strategy"] = "aggregated_profile"
                plots.append(plot_spec)

        else:
            # Default: return sampled raw points
            sample_df = df.sample(n=min(len(df), MAX_RAW_DATA_POINTS))
            plot_spec["data"] = sample_df.to_dict(orient="records")
            plot_spec["plot_strategy"] = "raw"
            plots.append(plot_spec)

    except Exception as e:
        logger.exception("Error during visualization generation: %s", e)

    return nl_response, plots


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
