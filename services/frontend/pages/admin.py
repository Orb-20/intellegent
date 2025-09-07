import streamlit as st
# --- CORRECTED IMPORT ---
from services.frontend import api_client

def render():
    """Renders the admin and diagnostics page."""
    st.header("Admin & Diagnostics")
    st.subheader("Backend Health Check")

    health = api_client.check_health()
    if health.get("status") == "ok":
        st.success(f"✅ Backend is healthy. Message: `{health.get('message', 'OK')}`")
    else:
        st.error(f"❌ Backend health check failed: `{health.get('message', 'N/A')}`")