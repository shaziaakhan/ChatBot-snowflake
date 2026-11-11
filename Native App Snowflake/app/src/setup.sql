-- Setup script for Chatbot Native App
CREATE APPLICATION ROLE IF NOT EXISTS chatbot_app_role;

CREATE OR ALTER VERSIONED SCHEMA code_schema;

GRANT USAGE ON SCHEMA code_schema TO APPLICATION ROLE chatbot_app_role;

-- Create or replace the Streamlit app object
CREATE OR REPLACE STREAMLIT code_schema.chatbot_streamlit
  FROM '/streamlit'
  MAIN_FILE = '/_Chatbot_App.py';

GRANT USAGE ON STREAMLIT code_schema.chatbot_streamlit TO APPLICATION ROLE chatbot_app_role;

-- Grant access to required data tables
GRANT USAGE ON DATABASE TEST_DB TO APPLICATION ROLE chatbot_app_role;
GRANT USAGE ON SCHEMA TEST_DB.PUBLIC TO APPLICATION ROLE chatbot_app_role;

GRANT SELECT ON TABLE TEST_DB.PUBLIC.SALES TO APPLICATION ROLE chatbot_app_role;
GRANT SELECT ON TABLE TEST_DB.PUBLIC.RETURN TO APPLICATION ROLE chatbot_app_role;
GRANT SELECT ON TABLE TEST_DB.PUBLIC.REPRESENTATIVES TO APPLICATION ROLE chatbot_app_role;
