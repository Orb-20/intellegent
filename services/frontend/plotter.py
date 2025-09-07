# services/frontend/plotter.py
from typing import List, Dict, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def _create_figure_from_spec(plot: Dict) -> Optional[go.Figure]:
    """Converts a single plot specification from the backend into a Plotly figure."""
    ptype = plot.get("plot_type", "").lower()
    title = plot.get("title", "Chart")
    data = plot.get("data", [])
    strategy = plot.get("plot_strategy", "raw")

    if not data:
        st.warning(f"Plot '{title}' received no data from the backend.")
        return None
    
    df = pd.DataFrame(data)
    if df.empty:
        return None

    try:
        # --- NEW: Handle aggregated profile plot strategy ---
        if strategy == "aggregated_profile" and ptype == "profile":
            x_col = plot.get("x")

            # Main line (mean)
            main_line = go.Scatter(
                x=df['mean'],
                y=df['pres_bin_mid'],
                mode='lines',
                name='Average'
            )
            # Upper bound (mean + std)
            error_band_upper = go.Scatter(
                x=df['mean'] + df['std'],
                y=df['pres_bin_mid'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
            # Lower bound (mean - std) with fill
            error_band_lower = go.Scatter(
                x=df['mean'] - df['std'],
                y=df['pres_bin_mid'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonextx',
                showlegend=True,
                name='Std. Dev.'
            )

            fig = go.Figure([error_band_lower, error_band_upper, main_line])
            fig.update_layout(title=title)
            fig.update_xaxes(title_text=x_col)
            fig.update_yaxes(title_text="Pressure (dbar)", autorange="reversed")
            return fig

        # --- Existing plot logic for raw data ---
        if ptype == "histogram":
            if "bin_start" in df.columns and "frequency" in df.columns:
                return px.bar(df, x="bin_start", y="frequency", title=title,
                              labels={"bin_start": plot.get("x", "Value"), "frequency": "Count"})
            else:
                return px.histogram(df, x=plot.get("x"), title=title, nbins=50)

        elif ptype == "bar_chart":
            return px.bar(df, x=plot.get("x"), y=plot.get("y"), title=title)

        elif ptype == "scatter_geo":
            return px.scatter_mapbox(
                df,
                lat=plot.get("lat", "latitude"),
                lon=plot.get("lon", "longitude"),
                hover_name=plot.get("label", "profile_id"),
                zoom=1,
                height=600,
                title=title,
                mapbox_style="carto-positron"
            )

        elif ptype == "timeseries":
            x_col = plot.get("x")
            if x_col and x_col in df.columns:
                df[x_col] = pd.to_datetime(df[x_col])
                return px.line(df, x=x_col, y=plot.get("y"), title=title, markers=True)
            else:
                st.warning(f"Timeseries plot '{title}' is missing its x-axis column '{x_col}'.")
                return None

        elif ptype == "scatter":
            return px.scatter(df, x=plot.get("x"), y=plot.get("y"), title=title)

        elif ptype == "profile":
            # Raw profile plot (small datasets only)
            x_col = plot.get("x")
            y_col = plot.get("y")
            fig = px.line(df, x=x_col, y=y_col, title=title,
                          labels={y_col: "Pressure (dbar)", x_col: "Value"})
            fig.update_yaxes(autorange="reversed")
            return fig

    except Exception as e:
        st.error(f"Could not render plot '{title}': {e}")
    return None

def render_plots(plots: List[Dict]):
    """Renders a list of plot specifications from the API response."""
    if not plots:
        return

    valid_figs = [fig for fig in [_create_figure_from_spec(p) for p in plots] if fig]

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
