# Import python packages
import streamlit as st
import plotly.express as px
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark import Session

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


# Get the current credentials
# session = get_active_session()

"""
Cortex Analyst App
====================
This app allows users to interact with their data using natural language.
"""

import json  # To handle JSON data
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import _snowflake  # For interacting with Snowflake-specific APIs
import pandas as pd
from snowflake.snowpark.exceptions import SnowparkSQLException

# List of available semantic model paths in the format: @<database>.<schema>.<stage>/<file>.yaml
AVAILABLE_SEMANTIC_MODELS_PATHS = [
    '@"TEST_DB"."PUBLIC"."MY_STAGE"/pizza.yaml'
]
API_ENDPOINT = "/api/v2/cortex/analyst/message"
API_TIMEOUT = 50000  # in milliseconds

# Initialize a Snowpark session for executing queries
session = get_active_session()


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        reset_session_state()
    show_header_and_sidebar()
    if len(st.session_state.messages) == 0:
        process_user_input("What questions can I ask?")
    display_conversation()
    handle_user_inputs()
    handle_error_notifications()


def reset_session_state():
    """Reset important session state elements."""
    st.session_state.messages = []  # List to store conversation messages
    st.session_state.active_suggestion = None  # Currently selected suggestion


def show_header_and_sidebar():
    """Display the header and sidebar of the app."""
    # Set the title and introductory text of the app
    st.title("Cortex Analyst")
    st.markdown(
        "Welcome to Cortex Analyst! Type your questions below to interact with your data."
    )

    # Sidebar with a reset button
    with st.sidebar:
        st.selectbox(
            "Selected semantic model:",
            AVAILABLE_SEMANTIC_MODELS_PATHS,
            format_func=lambda s: s.split("/")[-1],
            key="selected_semantic_model_path",
            on_change=reset_session_state,
        )
        st.divider()
        # Center this button
        _, btn_container, _ = st.columns([2, 6, 2])
        if btn_container.button("Clear Chat History", use_container_width=True):
            reset_session_state()


def handle_user_inputs():
    """Handle user inputs from the chat interface."""
    # Handle chat input
    user_input = st.chat_input("What is your question?")
    if user_input:
        process_user_input(user_input)
    # Handle suggested question click
    elif st.session_state.active_suggestion is not None:
        suggestion = st.session_state.active_suggestion
        st.session_state.active_suggestion = None
        process_user_input(suggestion)


def handle_error_notifications():
    if st.session_state.get("fire_API_error_notify"):
        st.toast("An API error has occurred!", icon="🚨")
        st.session_state["fire_API_error_notify"] = False


def process_user_input(prompt: str):
    """
    Process user input and update the conversation history.

    Args:
        prompt (str): The user's input.
    """

    # Create a new message, append to history and display immediately
    new_user_message = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }
    st.session_state.messages.append(new_user_message)
    with st.chat_message("user"):
        user_msg_index = len(st.session_state.messages) - 1
        display_message(new_user_message["content"], user_msg_index)

    # Show progress indicator inside analyst chat message while waiting for response
    with st.chat_message("analyst"):
        with st.spinner("Waiting for Analyst's response..."):
            time.sleep(1)
            response, error_msg = get_analyst_response(st.session_state.messages)
            if error_msg is None:
                analyst_message = {
                    "role": "analyst",
                    "content": response["message"]["content"],
                    "request_id": response["request_id"],
                }
            else:
                analyst_message = {
                    "role": "analyst",
                    "content": [{"type": "text", "text": error_msg}],
                    "request_id": response["request_id"],
                }
                st.session_state["fire_API_error_notify"] = True
            st.session_state.messages.append(analyst_message)
            st.rerun()


def get_analyst_response(messages: List[Dict]) -> Tuple[Dict, Optional[str]]:
    """
    Send chat history to the Cortex Analyst API and return the response.

    Args:
        messages (List[Dict]): The conversation history.

    Returns:
        Tuple[Optional[Dict], Optional[str]]: The response from the Cortex Analyst API.
    """
    # Prepare the request body with the user's prompt
    request_body = {
        "messages": messages,
        "semantic_model_file": f"{st.session_state.selected_semantic_model_path}",
    }

    # Send a POST request to the Cortex Analyst API endpoint
    resp = _snowflake.send_snow_api_request(
        "POST",  # method
        API_ENDPOINT,  # path
        {},  # headers
        {},  # params
        request_body,  # body
        None,  # request_guid
        API_TIMEOUT,  # timeout in milliseconds
    )

    # Content is a string with serialized JSON object
    parsed_content = json.loads(resp["content"])

    # Check if the response is successful
    if resp["status"] < 400:
        return parsed_content, None
    else:
        # Craft readable error message
        error_msg = f"""
        An Analyst API error has occurred 🚨

* Response code: `{resp['status']}`
* Request ID: `{parsed_content['request_id']}`
* Error code: `{parsed_content['error_code']}`

Message:
        """
        return parsed_content, error_msg


def display_conversation():
    """
    Display the conversation history between the user and the assistant.
    """
    for idx, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            display_message(content, idx)


def display_message(content: List[Dict[str, str]], message_index: int):
    """
    Display a single message content.

    Args:
        content (List[Dict[str, str]]): The message content.
        message_index (int): The index of the message.
    """
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            # Display suggestions as buttons
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                if st.button(
                    suggestion, key=f"suggestion_{message_index}_{suggestion_index}"
                ):
                    st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            # Display the SQL query and results
            display_sql_query(item["statement"], message_index)
        else:
            # Handle other content types if necessary
            pass


@st.cache_data(show_spinner=False)
def get_query_exec_result(query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute the SQL query, convert the results to a pandas DataFrame, and generate a general explanation of the results.

    Args:
        query (str): The SQL query.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: The query results and the error message.
    """
    global session
    try:
        df = session.sql(query).to_pandas()

        # Ensure the date column is parsed correctly
        if "date_time" in df.columns:  # Replace "date_time" with the actual column name
            try:
                # Parse the date column with the correct format
                df["date_time"] = pd.to_datetime(df["date_time"], format="%mm/%dd/%yyyy %H:%M", errors="coerce")
            except ValueError as e:
                return None, f"Error parsing date column: {str(e)}"

        return df, None
    except SnowparkSQLException as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def get_query_exec_result(query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute the SQL query and convert the results to a pandas DataFrame.

    Args:
        query (str): The SQL query.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: The query results and any error message.
    """
    global session
    try:
        df = session.sql(query).to_pandas()
        return df, None
    except SnowparkSQLException as e:
        return None, str(e)

def generate_query_explanation(sql: str) -> str:
    """
    Generate a general explanation of the SQL query.

    Args:
        sql (str): The SQL query.

    Returns:
        str: A human-readable explanation of what the query does.
    """
    explanation = "🧐 **SQL Query Explanation:**\n"

    # Basic query type detection
    if "SELECT" in sql.upper():
        explanation += "- This query is retrieving data from a table.\n"
        if "WHERE" in sql.upper():
            explanation += "- It includes filtering conditions using a `WHERE` clause.\n"
        if "GROUP BY" in sql.upper():
            explanation += "- The query is aggregating data based on specific columns using `GROUP BY`.\n"
        if "ORDER BY" in sql.upper():
            explanation += "- The results are being sorted based on the `ORDER BY` clause.\n"
        if "JOIN" in sql.upper():
            explanation += "- Multiple tables are being combined using a `JOIN` operation.\n"

    elif "INSERT" in sql.upper():
        explanation += "- This query is inserting new data into a table.\n"
    
    elif "UPDATE" in sql.upper():
        explanation += "- This query is modifying existing records in a table.\n"
    
    elif "DELETE" in sql.upper():
        explanation += "- This query is removing data from a table.\n"

    explanation += "\n✅ The SQL query is structured to retrieve and process relevant information based on the specified conditions."

    return explanation.strip()

def display_sql_query(sql: str, message_index: int):
    """
    Execute the SQL query and display the results in the form of a dataframe, charts, and an explanation.

    Args:
        sql (str): The SQL query.
        message_index (int): The index of the message.
    """

    # Display the SQL query
    with st.expander("SQL Query", expanded=False):
        st.code(sql, language="sql")

    # Display the results of the SQL query
    with st.expander("Results", expanded=True):
        with st.spinner("Running SQL..."):
            df, err_msg = get_query_exec_result(sql)
            if df is None:
                st.error(f"Could not execute generated SQL query. Error: {err_msg}")
                return

            if df.empty:
                st.write("Query returned no data")
                return

            # Show query results in three tabs
            data_tab, chart_tab, explanation_tab = st.tabs(["Data 📄", "Chart 📈", "Explanation 🧐"])

            with data_tab:
                st.dataframe(df, use_container_width=True)

            with chart_tab:
                display_charts_tab(df, message_index)

            with explanation_tab:
                query_explanation = generate_query_explanation(sql)
                st.markdown(query_explanation)

def display_charts_tab(df: pd.DataFrame, message_index: int) -> None:
    """
    Display the charts tab.

    Args:
        df (pd.DataFrame): The query results.
        message_index (int): The index of the message.
    """
    # There should be at least 2 columns to draw charts
    if len(df.columns) >= 2:
        all_cols_set = set(df.columns)
        col1, col2 = st.columns(2)
        x_col = col1.selectbox(
            "X axis", all_cols_set, key=f"x_col_select_{message_index}"
        )
        y_col = col2.selectbox(
            "Y axis",
            all_cols_set.difference({x_col}),
            key=f"y_col_select_{message_index}",
        )
        chart_type = st.selectbox(
            "Select chart type",
            options=["Line Chart 📈", "Bar Chart 📊", "Pie Chart 🥧"],
            key=f"chart_type_{message_index}",
        )
        if chart_type == "Line Chart 📈":
            st.line_chart(df.set_index(x_col)[y_col])
        elif chart_type == "Bar Chart 📊":
            st.bar_chart(df.set_index(x_col)[y_col])
        elif chart_type == "Pie Chart 🥧":
            # Ensure the Y-axis column is numeric for pie chart
            if pd.api.types.is_numeric_dtype(df[y_col]):
                st.plotly_chart(
                    px.pie(df, names=x_col, values=y_col, title="Pie Chart"),
                    use_container_width=True,
                )
            else:
                st.error("The selected Y-axis column must be numeric for a pie chart.")
    else:
        st.write("At least 2 columns are required")
if __name__ == "__main__":
    main()
