# services/frontend/api_client.py
import os
import requests
import streamlit as st

# --- API Configuration ---
API_URL = os.getenv("API_URL", "http://api:8000")
QUERY_ENDPOINT = f"{API_URL}/query"
HEALTH_ENDPOINT = f"{API_URL}/health"

@st.cache_data(ttl=3600, show_spinner="Querying backend...")
def query_backend(question: str, conversation_id: str | None) -> dict:
    """Calls the backend API with a natural language question."""
    payload = {"question": question, "conversation_id": conversation_id}
    try:
        r = requests.post(QUERY_ENDPOINT, json=payload, timeout=90)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: Could not connect to the backend at `{API_URL}`.")
        return {"error": str(e)}
    except ValueError:
        st.error("Invalid response from backend.")
        return {"error": "Invalid JSON response"}

def check_health() -> dict:
    """Checks the health of the backend API."""
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Failed to connect: {e}"}