#
import streamlit as st
import plotly.express as px
from snowflake.snowpark.context import get_active_session



# Get the current credentials
session = get_active_session()
session_id = session.session_id  # From snowflake.connector

import json  # To handle JSON data
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import _snowflake  # For interacting with Snowflake-specific APIs
import pandas as pd
from snowflake.snowpark.exceptions import SnowparkSQLException

# List of available semantic model paths in the format: @<database>.<schema>.<stage>/<file>.yaml
AVAILABLE_SEMANTIC_MODELS_PATHS = [
    '@"TEST_DB"."PUBLIC"."MY_STAGE"/global_sales.yaml',
    '@"TEST_DB"."PUBLIC"."MY_STAGE"/pizza.yaml'
    
]
API_ENDPOINT = "/api/v2/cortex/analyst/message"
API_TIMEOUT = 50000  # in milliseconds



def add_custom_css():
    custom_css = """
    <style>
    /* Set the overall background color for the app */
    body {
        background-color: #f4f7f6;
    }

    /* Style for the header */
    .streamlit-expanderHeader {
        font-size: 20px !important;
        font-weight: bold;
        color: #2a4d8d;
    }

    /* Style for the title */
    h1 {
        color: #2a4d8d;
        font-family: 'Roboto', sans-serif;
    }

    /* Customize the sidebar background */
    .css-1d391kg {
        background-color: #2a4d8d !important;
        color: white !important;
    }

    /* Style the buttons */
    .stButton > button {
        background-color: #CFEAE2;
        color: black;
        border-radius: 12px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: white !important;
        color: black !important;
        border: 2px solid #455F56;
    }
    
    section[data-testid="stSidebar"] .stDownloadButton > button {
        background-color: #CFEAE2 !important;
        color: black !important;
        border-radius: 12px !important;
        font-weight: bold !important;
        transition: background-color 0.3s ease !important;
        width: 100% !important;
        border: none !important;
        margin: 0.25rem 0 !important;
    }

    section[data-testid="stSidebar"] .stDownloadButton > button:hover {
        background-color: white !important;
        color: black !important;
        border: 2px solid #455F56 !important;
    }

    /* Add some padding between widgets */
    .stTextInput, .stSelectbox, .stButton {
        margin-top: 10px;
        margin-bottom: 10px;
    }

    /* Customize the chat bubble styles */
    .streamlit-chat-message {
        padding: 12px 18px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
    }


    /* Customize user message bubbles */
    .streamlit-chat-message[data-role="user"] {
        background-color: #00BFFF;
        color: white;
    }

   /* More generic selector for analyst messages */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"]) {
    border-left: 4px solid #1d3f8a;
    background-color: #f9f9f9;
    padding: 12px 18px;
    border-radius: 12px;
    margin-bottom: 10px;
}

    /* Style for section headers */
    .stExpanderHeader {
        background-color: #e3f2fd;
        color: #1c3b66;
        font-weight: bold; !important
    }

    /* Add custom background color to selected query result section */
    .stDataFrame {
        background-color: #CFEAE2;
        border-radius: 8px;
        padding: 10px;
    }
    
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)



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
    add_custom_css()



def reset_session_state():
    """Reset important session state elements."""
    st.session_state.messages = []  # List to store conversation messages
    st.session_state.active_suggestion = None  # Currently selected suggestion


def show_header_and_sidebar():
    """Display the header and sidebar of the app.🤖"""
    st.title("❄️ SalesBot ❄️")

    st.markdown(
        """
        
        **Ask. Analyze. Act.**  
        Get real-time sales insights with the power of *Cortex Analyst* under the hood.  
        One question away from unlocking your insights 💡
        """,
        unsafe_allow_html=True
    )


    with st.sidebar:
        st.selectbox(
            "Selected semantic model:",
            AVAILABLE_SEMANTIC_MODELS_PATHS,
            format_func=lambda s: s.split("/")[-1],
            key="selected_semantic_model_path",
            on_change=reset_session_state,
        )
        st.divider()

        # _, btn_container, _ = st.columns([2, 6, 2])
        if st.button("Clear Chat History", use_container_width=True):
            reset_session_state()

        # Only show "Save as PDF" if messages exist
        if st.session_state.messages:
            pdf_data = generate_chat_pdf(st.session_state.messages)
            st.download_button(
                label=" Save Chat (PDF)",
                data=pdf_data,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        if st.session_state.messages:
            if st.button("🧠 Smart Summary Insights", use_container_width=True):
                with st.spinner("Generating smart summary via Cortex..."):
                    summary = generate_smart_summary(st.session_state.messages)
                    if summary:
                        st.success("Smart Summary:")
                        st.markdown(summary)
                    else:
                        st.error("Failed to generate summary.")
                
           

        
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

* Response code: {resp['status']}
* Request ID: {parsed_content['request_id']}
* Error code: {parsed_content['error_code']}

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
            explanation += "- It includes filtering conditions using a WHERE clause.\n"
        if "GROUP BY" in sql.upper():
            explanation += "- The query is aggregating data based on specific columns using GROUP BY.\n"
        if "ORDER BY" in sql.upper():
            explanation += "- The results are being sorted based on the ORDER BY clause.\n"
        if "JOIN" in sql.upper():
            explanation += "- Multiple tables are being combined using a JOIN operation.\n"

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
            data_tab, chart_tab, explanation_tab = st.tabs(["DATA", "CHART", "EXPLANATION"])

            with data_tab:
                st.dataframe(df, use_container_width=True)

            with chart_tab:
                display_charts_tab(df, message_index)

            with explanation_tab:
                query_explanation = generate_query_explanation(sql)
                st.markdown(query_explanation)

import pandas as pd
import streamlit as st
import plotly.express as px

def display_charts_tab(df: pd.DataFrame, message_index: int) -> None:
    """
    Display the charts tab.

    Args:
        df (pd.DataFrame): The query results.
        message_index (int): The index of the message.
    """
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
            options=["Line Chart", "Bar Chart", "Pie Chart"],
            key=f"chart_type_{message_index}",
        )

        if chart_type == "Line Chart":
            st.line_chart(df.set_index(x_col)[y_col])
        elif chart_type == "Bar Chart":
            st.bar_chart(df.set_index(x_col)[y_col])
        elif chart_type == "Pie Chart":
            if pd.api.types.is_numeric_dtype(df[y_col]):
                st.plotly_chart(
                    px.pie(df, names=x_col, values=y_col, title="Pie Chart"),
                    use_container_width=True,
                )
            else:
                st.error("The selected Y-axis column must be numeric for a pie chart.")
        
    else:
        st.write("At least 2 columns are required")


from fpdf import FPDF
import io

class StyledPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)

    def add_message(self, role: str, text: str):
        if role.lower() == "user":
            self.set_text_color(0, 0, 255)  # Blue
            self.set_font("Arial", style="B", size=12)
            self.cell(0, 10, "User:", ln=True)
        elif role.lower() == "analyst":
            self.set_text_color(100, 100, 100)  # Gray
            self.set_font("Arial", style="B", size=12)
            self.cell(0, 10, "Analyst:", ln=True)
        
        self.set_font("Arial", size=12)
        self.set_text_color(0, 0, 0)  # Reset to black
        self.multi_cell(0, 10, text)
        self.ln(5)  # Add space after each message

def generate_chat_pdf(messages: List[Dict]) -> bytes:
    """Generate a styled PDF from the chat history."""
    pdf = StyledPDF()

    for msg in messages:
        role = msg["role"]
        for item in msg["content"]:
            if item["type"] == "text":
                pdf.add_message(role, item["text"])
            elif item["type"] == "sql":
                sql = item.get("statement", "")
                df, err = get_query_exec_result(sql)
                if df is not None and not df.empty:
                    sample = df.head(5).to_string(index=False)
                    pdf.add_message(role, f"Query Output (Top 5 rows):\n{sample}")
                elif err:
                    pdf.add_message(role, f"SQL Error: {err}")
    return bytes(pdf.output(dest="S").encode("latin1"))


def generate_smart_summary(messages: List[Dict]) -> Optional[str]:
    """
    Generate a smart summary of analyst messages and natural language summaries of SQL outputs.
    """
    from datetime import datetime

    def summarize_dataframe(df: pd.DataFrame) -> str:
        if df.empty:
            return "Query returned no data."

        summary_lines = []

        # Use column name and type heuristics
        num_cols = df.select_dtypes(include="number").columns
        str_cols = df.select_dtypes(include="object").columns

        if "total" in " ".join(col.lower() for col in df.columns):
            for col in num_cols:
                total = df[col].sum()
                summary_lines.append(f"Total {col.replace('_', ' ').title()}: {total:,.2f}")
        elif len(df.columns) == 2 and pd.api.types.is_numeric_dtype(df.dtypes[1]):
            summary_lines.append("Top breakdown:")
            for _, row in df.head(5).iterrows():
                key = row[0]
                value = row[1]
                summary_lines.append(f"- {key}: {value:,.2f}" if isinstance(value, (int, float)) else f"- {key}: {value}")
        elif "year" in df.columns.str.lower().tolist() and len(num_cols) == 1:
            metric = num_cols[0]
            summary_lines.append(f"Year-wise breakdown of {metric.replace('_', ' ').title()}:")
            for _, row in df.iterrows():
                summary_lines.append(f"- {int(row['YEAR'])}: {row[metric]:,.2f}")
        elif "product" in " ".join(col.lower() for col in df.columns):
            summary_lines.append("Sample of products:")
            for _, row in df.head(5).iterrows():
                summary_lines.append(f"- {row[0]}")
        else:
            summary_lines.append("Top 5 result rows:")
            for _, row in df.head(5).iterrows():
                summary_lines.append("- " + ", ".join(f"{col}: {val}" for col, val in row.items()))

        return "\n".join(summary_lines)

    analyst_texts = []
    sql_summaries = []

    for msg in messages:
        if msg["role"] == "analyst":
            for item in msg["content"]:
                if item["type"] == "text":
                    analyst_texts.append(item["text"])
                elif item["type"] == "sql":
                    sql = item.get("statement", "")
                    df, err = get_query_exec_result(sql)
                    if df is not None:
                        sql_summaries.append(summarize_dataframe(df))
                    elif err:
                        sql_summaries.append(f"SQL Error: {err}")

    final_summary = "**🧠 Smart Summary**\n\n"

    # if analyst_texts:
    #     final_summary += "**Analyst Insights:**\n"
    #     final_summary += "\n\n".join(analyst_texts)
    #     final_summary += "\n\n"

    if sql_summaries:
        final_summary += "**SQL Output Summaries:**\n"
        final_summary += "\n\n".join(sql_summaries)

    return final_summary.strip()



if __name__ == "__main__":
    main()
