import streamlit as st
import os
import subprocess 
import signal
import psutil
import time
from typing import Dict, Any
from configure_graph import invoke_our_graph

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

def display_chat_history():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def process_langgraph_response(response: Dict[str, Any]) -> str:
    """Extract the final content from LangGraph response."""
    try:
        messages = response["messages"]
        for message in reversed(messages):
            if (hasattr(message, 'content') and 
                not any(skip in message.content.lower() 
                        for skip in ['transfer', 'successfully transferred'])):
                return message.content
        return "Sorry, I couldn't process that request."
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")
        return "Sorry, I couldn't process that request."

def main():
    initialize_session_state()
    display_chat_history()

    if prompt := st.chat_input("Hello..."):
        if not st.session_state.processing:
            st.session_state.processing = True
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            try:
                with st.spinner("Thinking..."):
                    response = invoke_our_graph([prompt])
                    assistant_response = process_langgraph_response(response)
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                st.session_state.processing = False
                st.rerun()

if __name__ == "__main__":
    main()
