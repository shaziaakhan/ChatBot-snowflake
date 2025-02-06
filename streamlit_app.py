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

import pandas as pd

import streamlit as st

from snowflake.snowpark.context import get_active_session

from langchain.text_splitter import RecursiveCharacterTextSplitter
 
 
# Initialize Snowflake session
 
def fetch_data_in_chunks(query, rows_per_chunk, max_chunks):

    """Fetch data from Snowflake in manageable chunks."""

    for i in range(max_chunks):

        offset = i * rows_per_chunk

        chunk_query = f"{query} LIMIT {rows_per_chunk} OFFSET {offset}"

        chunk = session.sql(chunk_query).to_pandas()

        if chunk.empty:

            break

        yield chunk
 
def process_text_chunks(text):

    """Process text into chunks using LangChain's RecursiveCharacterTextSplitter."""

    text_splitter = RecursiveCharacterTextSplitter(

        separators=["\n"],  

        chunk_size=1500,  

        chunk_overlap=150,  

        length_function=len,

        add_start_index=True

    )

    return [chunk.page_content for chunk in text_splitter.create_documents([text])]
 
def preprocess_data_in_chunks(query):

    """Process entire dataset in chunks and return structured text chunks."""

    df = session.sql(query).to_pandas()

    total_rows = len(df)

    all_columns = set(df.columns)

    dataset_text = df.to_string(index=False)

    context = f"Row Count: {total_rows}\nColumns: {list(all_columns)}\nData:\n" + dataset_text

    return process_text_chunks(context)
 
def truncate_context(context, max_tokens=8192):

    """Truncate the context to fit within LLM token limits."""

    max_chars = max_tokens * 4  

    return context[:max_chars]
 
def ask_question():

    if "messages" not in st.session_state:

        st.session_state["messages"] = []

    for message in st.session_state["messages"]:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about the dataset:"):

        st.session_state["messages"].append({"role": "user", "content": question})

        with st.chat_message("user"):

            st.markdown(question)
 
        with st.chat_message("assistant"):

            message_placeholder = st.empty()

            with st.spinner(f"{st.session_state.selected_llm} is processing..."):

                try:

                    query = "SELECT * FROM TEST_DB.PUBLIC.PIZZA"

                    rows_per_chunk = st.session_state.rows_per_chunk

                    max_chunks = st.session_state.num_chunks

                    processed_chunks = preprocess_data_in_chunks(query)

                    summarized_context = "\n".join(processed_chunks)

                    truncated_context = truncate_context(summarized_context)

                    prompt = f"""

                    You are an expert system. Using the PREPROCESSED DATA provided below, answer the QUESTION concisely with only the final answer.

                    PREPROCESSED DATA:

                    {truncated_context}

                    QUESTION:

                    {question}

                    """

                    response_query = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS RESPONSE"

                    response = session.sql(response_query, params=[st.session_state.selected_llm, prompt]).collect()

                    res_text = response[0].RESPONSE

                    message_placeholder.markdown(res_text)

                    st.session_state["messages"].append({"role": "assistant", "content": res_text})

                except Exception as e:

                    message_placeholder.markdown(f"Error: {e}")
 
with st.sidebar:

    st.session_state.selected_llm = st.selectbox("Select the LLM you want to use:", [

       'claude-3-5-sonnet', 'llama3.2-3b', 'mixtral-8x7b', 'snowflake-arctic', 'mistral-large',

        'llama3-8b', 'llama3-70b', 'reka-flash', 'mistral-7b',

        'llama2-70b-chat', 'gemma-7b'

    ])

    st.session_state.rows_per_chunk = st.slider("Rows per chunk", min_value=100, max_value=5000, value=1000, step=100)

    st.session_state.num_chunks = st.slider("Number of chunks to process", min_value=1, max_value=50, value=10)
 
ask_question()

 
 
