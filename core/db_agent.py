import os
import sys
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
from sqlalchemy import create_engine

# Ensure parent directory is in the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import settings, get_llm
from utils.sanitiser import validate_sql_query
from utils.schema_extractor import get_db_schema

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)

class SecureQuerySQLDatabaseTool(QuerySQLDataBaseTool):
    """
    A secured SQL database query execution tool that validates queries
    via sanitiser.py before executing them on the database.
    """
    def _run(self, query: str, *args: Any, **kwargs: Any) -> str:
        try:
            # 1. Strip out markdown code blocks if the agent wrapped the query
            clean_query = query.strip()
            if clean_query.startswith("```sql"):
                clean_query = clean_query[6:]
            if clean_query.startswith("```"):
                clean_query = clean_query[3:]
            if clean_query.endswith("```"):
                clean_query = clean_query[:-3]
            clean_query = clean_query.strip()
            
            # 2. Run query through sanitiser validation
            validated_query = validate_sql_query(clean_query)
            
            # 3. Execute
            return super()._run(validated_query, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Blocked SQL execution attempt: '{query}' | Reason: {str(e)}")
            return f"Error / Blocked: {str(e)}"

class DBAgent:
    """
    Manages the connection to MySQL and the initialization of the LangChain SQL Agent.
    """
    def __init__(self, host: str = settings.DB_HOST, port: int = settings.DB_PORT, user: str = settings.DB_USER, password: str = settings.DB_PASSWORD, database: str = settings.DB_NAME):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.db = None
        self.agent_executor = None
        self.connection_error = None
        self._init_connection()

    def _init_connection(self):
        """
        Initialises connection to MySQL database.
        """
        if not self.database:
            self.connection_error = "Database name is not configured."
            return

        try:
            # Create connection string
            uri = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
            
            # Verify basic connection works before wrapping with LangChain
            engine = create_engine(uri, connect_args={"connect_timeout": 5})
            with engine.connect() as conn:
                pass
            
            # Wrap database
            self.db = SQLDatabase.from_uri(uri)
            logger.info("Successfully connected to MySQL database.")
            self.connection_error = None
            
            # Create the SQL Agent
            self._create_agent()
        except Exception as e:
            self.db = None
            self.agent_executor = None
            self.connection_error = f"Database Connection Error: {str(e)}"
            logger.error(f"Failed to connect to database: {e}")

    def _create_agent(self):
        """
        Creates a LangChain SQL Agent with secure query execution tools.
        """
        try:
            llm = get_llm(temperature=0.0)
            
            # Set up the toolkit
            toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)
            tools = toolkit.get_tools()
            
            # Replace QuerySQLDataBaseTool with SecureQuerySQLDatabaseTool
            for idx, tool in enumerate(tools):
                if isinstance(tool, QuerySQLDataBaseTool):
                    tools[idx] = SecureQuerySQLDatabaseTool(
                        db=self.db, 
                        description=tool.description
                    )
            
            # Load dynamic schema reference for the prompt context
            schema_info = get_db_schema(self.db._engine)
            
            # Construct system prompt
            system_prompt = (
                "You are a Senior BI Analyst and SQL Expert.\n"
                "Your objective is to answer user queries by retrieving data from the MySQL database.\n\n"
                "CRITICAL SECURITY CONSTRAINTS:\n"
                "1. You are ONLY allowed to execute read-only queries (SELECT, SHOW, DESCRIBE, EXPLAIN, WITH).\n"
                "2. Under NO circumstances should you perform any write, insert, update, delete, drop, alter, "
                "or configuration actions (e.g. changing DB variables). If the query calls for modifications, "
                "refuse the command and state that write actions are disabled.\n"
                "3. If a tool returns an 'Execution Blocked' error, explain the issue clearly to the user without attempting to bypass it.\n\n"
                "DATABASE DETAILS:\n"
                f"{schema_info}\n\n"
                "GUIDELINES:\n"
                "- Query only the columns that are needed.\n"
                "- Write correct, ANSI-compliant SQL queries.\n"
                "- When you obtain the final answer, summarise the findings clearly."
            )
            
            # Create agent
            # We use the standard React / Tool-calling agent configuration
            self.agent_executor = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                agent_type="openai-tools" if settings.LLM_PROVIDER == "openai" else "tool-calling",
                verbose=True,
                extra_tools=tools, # Explicitly pass the modified toolset to take precedence
                prefix=system_prompt
            )
            logger.info("LangChain SQL Agent successfully initialized.")
        except Exception as e:
            self.agent_executor = None
            logger.exception("Failed to create SQL Agent")

    def run_query_df(self, sql_query: str) -> pd.DataFrame:
        """
        Runs a direct SQL query and returns the output as a Pandas DataFrame.
        This bypasses the agent but still applies sanitiser security checks.
        Useful for retrieving the raw result set for charting.
        """
        if not self.db:
            raise ValueError(f"Database not connected. {self.connection_error or ''}")
            
        # Ensure SQL is safe
        safe_query = validate_sql_query(sql_query)
        
        # Run using pandas
        engine = self.db._engine
        return pd.read_sql(safe_query, engine)

    def ask_agent(self, query: str) -> Dict[str, Any]:
        """
        Passes a user query to the SQL agent and returns the text summary and raw table if applicable.
        """
        if self.connection_error:
            return {
                "answer": f"Database connection error: {self.connection_error}",
                "query": "",
                "dataframe": None
            }

        if not self.agent_executor:
            return {
                "answer": "SQL Agent is not initialized.",
                "query": "",
                "dataframe": None
            }

        try:
            # Execute agent query
            # We fetch response
            response = self.agent_executor.invoke({"input": query})
            answer = response.get("output", "No response generated.")
            
            # Try to extract the SQL query that was successfully run (for charting/rendering)
            # We can scan the history or query cache if possible.
            # As a fallback, we check if the agent outputted any SQL.
            # To make charting robust, we can run a quick LLM extraction on the agent execution logs,
            # or parse it. A simpler, robust way is to ask the LLM to extract the final SELECT query
            # from the agent run if we want to retrieve the actual dataframe.
            # Let's write a small extractor to get the SQL query from the response if any.
            sql_query = self._extract_sql(answer, query)
            
            df = None
            if sql_query:
                try:
                    df = self.run_query_df(sql_query)
                except Exception as e:
                    logger.warning(f"Could not load DataFrame for generated SQL '{sql_query}': {e}")
            
            return {
                "answer": answer,
                "query": sql_query,
                "dataframe": df
            }

        except Exception as e:
            logger.exception("Error executing database agent request")
            return {
                "answer": f"Error running database agent: {str(e)}",
                "query": "",
                "dataframe": None
            }

    def _extract_sql(self, text: str, user_query: str) -> str:
        """
        Utility to extract a valid SQL SELECT statement from the text response or query,
        or asks the LLM to extract the SQL query that was executed by the agent to construct the answer.
        """
        # Look for SQL blocks in response first
        sql_blocks = re.findall(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if sql_blocks:
            return sql_blocks[0].strip()
            
        sql_generic = re.findall(r"```\s*(SELECT.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if sql_generic:
            return sql_generic[0].strip()

        # Ask LLM to extract the SELECT query that answers the user's question,
        # or construct one directly if it wasn't printed.
        try:
            llm = get_llm(temperature=0.0)
            schema_info = get_db_schema(self.db._engine)
            
            system_prompt = (
                "You are an assistant that extracts or reconstructs the single SELECT SQL query "
                "used to retrieve the database values discussed in the context.\n"
                "If no SELECT query is mentioned or relevant, reply with 'NONE'.\n"
                "DO NOT write any comments, explanation, or wrap it in formatting. Just output the query.\n\n"
                f"Schema Context:\n{schema_info}\n\n"
                f"Conversation context:\n{text}"
            )
            
            messages = [
                ("system", system_prompt),
                ("user", f"Extract the SQL query that answers: {user_query}")
            ]
            
            extracted = llm.invoke(messages).content.strip()
            if "select" in extracted.lower():
                # Strip wrapping if the LLM output format included it
                if extracted.startswith("```sql"):
                    extracted = extracted[6:]
                if extracted.endswith("```"):
                    extracted = extracted[:-3]
                return extracted.strip()
        except Exception as e:
            logger.error(f"Failed to extract SQL query: {e}")
            
        return ""
import re
