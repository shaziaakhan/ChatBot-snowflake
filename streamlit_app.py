import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
# Set up the Streamlit page
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# Initialize Snowflake session
# session = get_active_session()
from snowflake.snowpark import Session
import streamlit as st

# Snowflake credentials (from secrets.toml or environment variables)
sf_options = {
    "account": st.secrets["snowflake"]["account"],
    "user": st.secrets["snowflake"]["user"],
    "password": st.secrets["snowflake"]["password"],
    "warehouse": st.secrets["snowflake"]["warehouse"],
    "database": st.secrets["snowflake"]["database"],
    "schema": st.secrets["snowflake"]["schema"],
}

# Initialize the Snowflake session
session = Session.builder.configs(sf_options).create()

# Now you can use the session to query Snowflake

# Sidebar options
with st.sidebar:
    st.session_state.selected_llm = st.selectbox(
        "Select the LLM you want to use:",
        [
            'llama3.2-3b',
            'llama3.1-8b',
            'jamba-instruct',
            'jamba-1.5-mini',
            'mixtral-8x7b',
            'snowflake-arctic',
            'mistral-large',
            'llama3-8b',
            'llama3-70b',
            'reka-flash',
            'mistral-7b',
            'llama2-70b-chat',
            'gemma-7b'
        ]
    )
    st.slider("Number of rows to fetch", min_value=1, max_value=1024, key="num_rows")
    st.slider("Number of messages to remember while answering question", 0, 10, key="slide_window")
# Function to process questions and return answers from the dataset
def ask_question():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    # Display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    # Get the user query
    if question := st.chat_input("Ask a question about the dataset:"):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        # Fetch relevant rows from the dataset
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner(f"{st.session_state.selected_llm} is processing..."):
                # Query to fetch data from the Snowflake table
                query = f"""
                    SELECT *
                    FROM TEST_DB.PUBLIC.GLOBAL_SALES
                    LIMIT ?
                """
                # Retrieve the most relevant rows
                data_df = session.sql(query, params=[st.session_state.num_rows]).to_pandas()
                context = data_df.to_string(index=False)
                # Build prompt
                prompt = f"""
                    You are an expert system answering questions based on the dataset provided below. Use the CONTEXT to answer the QUESTION accurately and concisely.
                    CONTEXT: {context}
                    QUESTION: {question}
                """
                # Query the LLM for an answer
                response_query = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS RESPONSE"
                response = session.sql(response_query, params=[st.session_state.selected_llm, prompt]).collect()
                # Display the response
                answer = response[0].RESPONSE.strip("'")
                message_placeholder.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})
# Main app execution
ask_question()
