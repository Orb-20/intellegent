import streamlit as st
# --- CORRECTED IMPORTS ---
from services.frontend import api_client
from services.frontend import plotter

def render():
    """Renders the main conversational chat interface."""
    st.header("ARGO Float Conversational Explorer")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you explore ARGO data today?"}]
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sql_query" in message:
                with st.expander("Show Generated SQL"):
                    st.code(message["sql_query"], language="sql")
            if message.get("plots"):
                plotter.render_plots(message["plots"])
            if message.get("follow_ups"):
                for followup in message["follow_ups"]:
                    # Use a more unique key for each button
                    if st.button(followup["text"], key=f"followup_{message['content']}_{followup['id']}"):
                         # Trigger a rerun with the follow-up question
                         st.session_state.last_prompt = followup["text"]
                         st.rerun()

    # Handle chat input
    prompt = st.chat_input("e.g., What is the average temperature in March 2022?")
    
    # Check if a follow-up button was clicked
    if "last_prompt" in st.session_state and st.session_state.last_prompt:
        prompt = st.session_state.last_prompt
        st.session_state.last_prompt = None # Reset after use

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Call backend and update state
        with st.spinner("Thinking..."):
            response = api_client.query_backend(prompt, st.session_state.conversation_id)
        
        if not response.get("error"):
            st.session_state.conversation_id = response.get("conversation_id")
            assistant_message = {
                "role": "assistant",
                "content": response.get("natural_language_response", "Here are the results:"),
                "sql_query": response.get("sql_query"),
                "plots": response.get("plots"),
                "follow_ups": response.get("follow_ups")
            }
            st.session_state.messages.append(assistant_message)
        else:
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {response['error']}"})
        
        st.rerun()