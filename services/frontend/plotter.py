# services/frontend/plotter.py
from typing import List, Dict, Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def _create_figure_from_spec(plot: Dict) -> Optional[go.Figure]:
    ptype = plot.get("plot_type", "").lower()
    title = plot.get("title", "Chart")
    data = plot.get("data", [])

    if not data:
        st.warning(f"No data was provided for the plot '{title}'.")
        return None
    
    df = pd.DataFrame(data)
    if df.empty:
        return None

    try:
        # STRATEGY 1: Render the new, multi-layered analytical profile plot.
        if ptype == "analytical_profile":
            x_label = plot.get("x_measure", "Value")
            fig = go.Figure()

            # Layer 1: Faint lines for the absolute min/max range.
            fig.add_trace(go.Scatter(
                x=df['max_val'], y=df['pres_bin'], mode='lines', line_color='rgba(173, 216, 230, 0.5)',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df['min_val'], y=df['pres_bin'], mode='lines', line_color='rgba(173, 216, 230, 0.5)',
                fillcolor='rgba(173, 216, 230, 0.2)', fill='tonextx', name='Min/Max Range'
            ))

            # Layer 2: Shaded area for Standard Deviation.
            fig.add_trace(go.Scatter(
                x=df['mean'] + df['std'], y=df['pres_bin'], mode='lines', line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df['mean'] - df['std'], y=df['pres_bin'], mode='lines', line_color='rgba(0,0,0,0)',
                fillcolor='rgba(78, 151, 211, 0.4)', fill='tonextx', name='Std. Deviation'
            ))
            
            # Layer 3: Bold line for the Average trend.
            fig.add_trace(go.Scatter(
                x=df['mean'], y=df['pres_bin'], mode='lines', name='Average', 
                line=dict(color='#003366', width=3)
            ))
            
            fig.update_layout(
                title_text=title, yaxis_title="Pressure (dbar)", xaxis_title=x_label,
                legend=dict(x=0.01, y=0.99, bordercolor="lightgrey", borderwidth=1)
            )
            fig.update_yaxes(autorange="reversed")
            return fig

        # STRATEGY 2: Fallback for simple, raw data plots.
        if ptype == "scatter_geo":
            return px.scatter_mapbox(
                df, lat=plot.get("lat"), lon=plot.get("lon"), hover_name=plot.get("label"),
                zoom=1, height=600, title=title, mapbox_style="carto-positron"
            )

    except Exception as e:
        st.error(f"Could not render plot '{title}'. Error: {e}")
    return None

def render_plots(plots: List[Dict]):
    if not plots:
        return
    
    valid_figs = [fig for fig in [_create_figure_from_spec(p) for p in plots] if fig]

    if not valid_figs:
        st.info("No valid visualizations could be generated for this query.")
        return

    if len(valid_figs) == 1:
        st.plotly_chart(valid_figs[0], use_container_width=True)
    else:
        tabs = st.tabs([f"📊 {fig.layout.title.text or f'Plot {i+1}'}" for i, fig in enumerate(valid_figs)])
        for i, tab in enumerate(tabs):
            with tab:
                st.plotly_chart(valid_figs[i], use_container_width=True)