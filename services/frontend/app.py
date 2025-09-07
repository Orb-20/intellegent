import streamlit as st
# --- CORRECTED IMPORTS ---
from services.frontend.pages import chat, explore, profiles, admin

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="FloatChat — ARGO Explorer",
        page_icon="🌊",
        layout="wide",
    )

    with st.sidebar:
        st.title("FloatChat")
        st.markdown("AI-driven ARGO Explorer")
        st.markdown("---")

        page_options = {
            "💬 Chat": chat.render,
            "🗺️ Geospatial Explorer": explore.render,
            "📈 Profile Comparison": profiles.render,

            "⚙️ Admin": admin.render
        }
        page_selection = st.radio("Navigation", list(page_options.keys()), label_visibility="collapsed")
        st.markdown("---")

    # Render the selected page
    page_options[page_selection]()

if __name__ == "__main__":
    main()