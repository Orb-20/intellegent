import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from services.frontend import api_client, plotter

def render():
    """Renders the map-based data exploration tool."""
    st.header("Geospatial Explorer")
    st.markdown("Filter ARGO profiles by location and date range.")

    with st.form("explore_form"):
        st.subheader("Filters")
        c1, c2 = st.columns(2)
        with c1:
            lat_center = st.number_input("Center Latitude", value=10.0, format="%.2f")
            lon_center = st.number_input("Center Longitude", value=70.0, format="%.2f")
            start_date = st.date_input("Start Date", datetime.date(2022, 3, 1))
        with c2:
            lat_window = st.slider("Latitude Window (±°)", 0.1, 10.0, 1.0, 0.1)
            lon_window = st.slider("Longitude Window (±°)", 0.1, 10.0, 1.0, 0.1)
            end_date = st.date_input("End Date", datetime.date(2022, 3, 31))

        submitted = st.form_submit_button("Search Area")

    if submitted:
        nl_prompt = (
            f"List profiles between lat {lat_center - lat_window:.2f} and {lat_center + lat_window:.2f}, "
            f"and lon {lon_center - lon_window:.2f} and {lon_center + lon_window:.2f}, "
            f"from {start_date.isoformat()} to {end_date.isoformat()}. "
            f"Return profile_id, juld, latitude, longitude. Limit to 500."
        )
        response = api_client.query_backend(nl_prompt, st.session_state.get("conversation_id"))
        
        if response.get("error"):
            st.error(response['error'])
        else:
            df = pd.DataFrame(response.get("data", []))
            if df.empty:
                st.warning("No profiles found matching your criteria.")
            else:
                st.success(f"Found {len(df)} profiles.")
                fig = px.scatter_mapbox(
                    df, lat="latitude", lon="longitude", hover_name="profile_id",
                    zoom=3, height=600, center={"lat": lat_center, "lon": lon_center}
                )
                fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
                plotter.render_plots(response.get("plots", []))