# services/frontend/plotter.py
from typing import List, Dict, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _create_figure_from_spec(plot: Dict) -> Optional[go.Figure]:
    """Converts a single plot specification from the backend into a Plotly figure."""
    ptype = plot.get("type", "").lower()
    title = plot.get("title", "Chart")
    
    try:
        if ptype == "timeseries" and plot.get("x") and plot.get("y"):
            df = pd.DataFrame({'time': pd.to_datetime(plot["x"]), 'value': plot["y"]})
            fig = px.line(df, x='time', y='value', title=title, markers=True)
            fig.update_layout(xaxis_title="Time", yaxis_title=plot.get("y_label", "Value"))
            return fig
            
        elif ptype == "scattergeo" and plot.get("lat") and plot.get("lon"):
            df = pd.DataFrame({'lat': plot["lat"], 'lon': plot["lon"], 'label': plot.get("label", "")})
            fig = px.scatter_mapbox(df, lat="lat", lon="lon", hover_name="label", zoom=1, height=600)
            fig.update_layout(mapbox_style="carto-positron", title=title)
            return fig
            
        elif ptype == "histogram" and plot.get("values"):
            return px.histogram(x=plot["values"], title=title, nbins=plot.get("bins", 30))
            
        # Add other plot types here as needed (boxplot, scatter_matrix, etc.)
            
    except Exception as e:
        st.warning(f"Could not render plot '{title}': {e}")
    return None

def render_plots(plots: List[Dict]):
    """Renders a list of plot specifications from the API response."""
    if not plots:
        return

    figs = [_create_figure_from_spec(p) for p in plots]
    valid_figs = [fig for fig in figs if fig]

    if not valid_figs:
        return

    if len(valid_figs) == 1:
        st.plotly_chart(valid_figs[0], use_container_width=True)
    else:
        tab_titles = [f"📊 {fig.layout.title.text or f'Plot {i+1}'}" for i, fig in enumerate(valid_figs)]
        tabs = st.tabs(tab_titles)
        for i, tab in enumerate(tabs):
            with tab:
                st.plotly_chart(valid_figs[i], use_container_width=True)