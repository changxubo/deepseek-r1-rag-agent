import os
 

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from retrieval_graph.graph import invoke_graph
from utils import get_streamlit_cb  # Utility function to get a Streamlit callback handler with context
import asyncio
 
os.environ["NVIDIA_API_KEY"] = "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"
st.title("RAG Reseach Agent with Deepseek-R1")
st.markdown("# Deepseek-R1 RAG Agent Full Implementation")
# st write magic
"""
    This is a simple RAG agent that uses langchain,langgraph,streamlit and NVIDIA NIM APIs to implement a deep research agent.
"""

if "messages" not in st.session_state:
    # default initial message to render in message state
    st.session_state["messages"] = [AIMessage(content="How can I help you?")]

# Loop through all messages in the session state and render them as a chat on every st.refresh mech
for msg in st.session_state.messages:
    # https://docs.streamlit.io/develop/api-reference/chat/st.chat_message
    # we store them as AIMessage and HumanMessage as its easier to send to LangGraph
    if type(msg) == AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) == HumanMessage:
        st.chat_message("user").write(msg.content)

# takes new input in chat box from user and invokes the graph
if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # Process the AI's response and handles graph events using the callback mechanism
    with st.chat_message("assistant"):
        st_placeholder = st.container()
        st_messages = st.session_state.messages
        respone = asyncio.run(invoke_graph(st_messages, st_placeholder))
        st.session_state.messages.append(AIMessage(content=respone))  