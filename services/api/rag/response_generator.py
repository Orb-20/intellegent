# services/api/rag/response_generator.py
import json
import logging
from typing import Any, List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from . import llm_utils

logger = logging.getLogger(__name__)

def df_to_json_rows(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.replace({np.nan: None}).to_dict(orient="records")

def _decide_visualizations(question: str, df: pd.DataFrame) -> List[Dict]:
    if df.empty:
        return []

    plots = []
    
    if df.attrs.get('is_analytical'):
        plots.append({
            "plot_type": "analytical_profile",
            "x_measure": df.attrs.get('x_measure', 'value'),
            "title": llm_utils.run_plot_title_generation(question) or "Analytical Profile Plot",
            "data": df_to_json_rows(df),
        })
        return plots

    cols = set(df.columns)
    if {'latitude', 'longitude'}.issubset(cols):
        plots.append({
            "plot_type": "scatter_geo", "lat": "latitude", "lon": "longitude",
            "label": "profile_id", "title": "Geospatial Data",
            "data": df_to_json_rows(df.sample(n=min(len(df), 2000))),
        })
    
    return plots

def generate_final_response(
    question: str, df: pd.DataFrame, sql: str,
    provenance: List[Dict], diagnostics: List[str]
) -> Tuple[str, List[Dict]]:
    
    nl_response = f"Your query returned {len(df)} results."
    data_summary = f"The query produced a table with {len(df)} rows and the following columns: {df.columns.tolist()}."

    if df.attrs.get('is_analytical') and not df.empty:
        total_points = df['count'].sum()
        surface_bin = df.iloc[0]
        deep_bin = df.iloc[-1]
        data_summary = (
            f"An analysis was performed on {total_points:,.0f} data points, binned by depth. "
            f"At the surface (around {surface_bin['pres_bin']} dbar), the average temperature is {surface_bin['mean']:.2f}°C, with a range from {surface_bin['min_val']:.2f} to {surface_bin['max_val']:.2f}°C. "
            f"In the deepest analyzed layer (around {deep_bin['pres_bin']} dbar), the average is {deep_bin['mean']:.2f}°C, ranging from {deep_bin['min_val']:.2f} to {deep_bin['max_val']:.2f}°C."
        )
        nl_response = "I have analyzed the relationship for you."

    try:
        # Use llm_utils, not the mistyped ll_utils <-- CORRECTED TYPO
        generated_response = llm_utils.run_answer_generation(context=data_summary, question=question)
        if generated_response:
            nl_response = generated_response
    except Exception as e:
        logger.exception("Error during answer generation: %s", e)

    plots = _decide_visualizations(question, df)
    if plots:
        diagnostics.append(f"Generated {len(plots)} plot(s) using the rule-based engine.")
        
    return nl_response, plots