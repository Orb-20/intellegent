import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from services.frontend import api_client, plotter

def render():
    """Renders the tool for comparing measurement profiles."""
    st.header("Profile Comparison")
    st.markdown("Visualize and compare vertical profiles (temperature, salinity) for specific profile IDs.")
    profile_input = st.text_input("Enter Profile IDs (comma-separated)", "15, 23")

    if st.button("Load and Compare Profiles"):
        ids = [p.strip() for p in profile_input.split(",") if p.strip().isdigit()]
        if not ids:
            st.warning("Please enter at least one valid profile ID.")
        else:
            nl_prompt = (
                f"Fetch level data for profile_ids {', '.join(ids)}. "
                f"Return profile_id, pres_dbar, temp_degc, psal_psu. "
                f"Order by profile_id, pres_dbar DESC."
            )
            response = api_client.query_backend(nl_prompt, st.session_state.get("conversation_id"))
            if response.get("error"):
                st.error(response['error'])
            else:
                df = pd.DataFrame(response.get("data", []))
                if df.empty:
                    st.warning("No data found for the specified profile IDs.")
                else:
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Temperature vs Pressure", "Salinity vs Pressure"))
                    for pid, g in df.groupby("profile_id"):
                        g = g.sort_values("pres_dbar", ascending=False)
                        fig.add_trace(go.Scatter(x=g["temp_degc"], y=g["pres_dbar"], name=f"Temp {pid}", mode='lines'), row=1, col=1)
                        fig.add_trace(go.Scatter(x=g["psal_psu"], y=g["pres_dbar"], name=f"Salinity {pid}", mode='lines'), row=1, col=2)
                    
                    fig.update_yaxes(title_text="Pressure (dbar)", autorange="reversed", row=1, col=1)
                    fig.update_yaxes(autorange="reversed", row=1, col=2)
                    fig.update_layout(height=600, legend_title_text='Profile ID')
                    st.plotly_chart(fig, use_container_width=True)
                    plotter.render_plots(response.get("plots", []))