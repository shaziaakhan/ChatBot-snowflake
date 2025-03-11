# ChatBot-Snowflake

## Overview
ChatBot-Snowflake is an advanced AI-powered chatbot designed to seamlessly interact with Snowflake databases, enabling intelligent data exploration, analysis, and visualization through natural language queries. This project leverages cutting-edge AI models and cloud computing capabilities to provide real-time insights from structured datasets.

The system comprises two Streamlit applications:
1. **AI-Driven Chatbot for Dataset Interaction** – An intelligent assistant that allows users to query structured data using natural language and receive AI-generated insights.
2. **Cortex Analyst for Interactive Data Visualization** – A powerful visualization tool that integrates with Snowflake Cortex to provide data-driven insights through interactive plots and analytics.

## Key Features

### Application 1: AI-Powered Chatbot for Dataset Interaction
- **Conversational AI Interface** – Enables users to query datasets using natural language.
- **Snowflake Snowpark Integration** – Retrieves and preprocesses large datasets with high efficiency.
- **LangChain-Powered Processing** – Uses `RecursiveCharacterTextSplitter` for structured data handling.
- **AI-Driven Response Generation** – Utilizes Snowflake Cortex LLMs for intelligent data-driven responses.
- **Multi-Model AI Support** – Allows switching between advanced models such as `claude-3-5-sonnet`, `mixtral-8x7b`, and `snowflake-arctic`.

### Application 2: Cortex Analyst for Data Visualization
- **Dynamic Data Retrieval** – Connects to Snowflake databases to fetch and analyze structured data.
- **AI-Powered Data Interpretation** – Leverages Cortex Analyst to generate insights from complex datasets.
- **Interactive Visualizations** – Uses Plotly for generating intuitive and interactive data charts.
- **Conversation Memory & Context Awareness** – Retains session history for a more intuitive user experience.
- **Custom Semantic Model Selection** – Allows users to choose from predefined model paths for optimized performance.

## Installation & Setup

### Prerequisites
Ensure the following dependencies are installed before running the chatbot:
- Python 3.8+
- Streamlit
- Snowflake Connector for Python
- LangChain
- Pandas
- Plotly
- A `secrets.toml` file for Snowflake credentials

### Installation Steps
1. Clone the repository:
 ```bash
   git clone <repo-url>
   cd <repo-directory>
 ```
2. Create a virtual environment and activate it:
   
```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
```
3. Install dependencies:
   
```bash
   pip install -r requirements.txt
```

4. Set up secrets.toml for Snowflake credentials:

```bash

toml
   [snowflake]
   account = "your_snowflake_account"
   user = "your_username"
   password = "your_password"
   warehouse = "your_warehouse"
   database = "your_database"
   schema = "your_schema"
```
5. Run the Streamlit app:
   
```bash
   streamlit run streamlit_app.py  # For AI-powered dataset interaction
   streamlit run cortex_analyst.py  # For Cortex Analyst and visualization
```
