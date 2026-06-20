import os
import sys
import logging
import pandas as pd
import streamlit as st
from langchain_groq import ChatGroq

# Ensure parent directory is in the path
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import local components
from config import settings, get_llm
from utils.sanitiser import clean_natural_language_input, SQLSecurityException, PromptInjectionException
from utils.chart_engine import generate_auto_chart
from core.router import route_query
from core.rag_chain import RAGPipeline
from sqlalchemy import create_engine
from core.db_agent import DBAgent

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Streamlit Layout & Premium Styling ---
st.set_page_config(
    page_title="Enterprise BI AI Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium styles using Markdown custom CSS injections
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #1E293B;
    }
    
    .status-card {
        border-radius: 12px;
        padding: 16px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    .badge-connected {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-disconnected {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialisation ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load settings from st.secrets if they exist (production configuration)
try:
    if "GROQ_API_KEY" in st.secrets:
        settings.GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    if "OPENAI_API_KEY" in st.secrets:
        settings.OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    if "ANTHROPIC_API_KEY" in st.secrets:
        settings.ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
        
    if "DB_HOST" in st.secrets:
        settings.DB_HOST = st.secrets["DB_HOST"]
    if "DB_USER" in st.secrets:
        settings.DB_USER = st.secrets["DB_USER"]
    if "DB_PASSWORD" in st.secrets:
        settings.DB_PASSWORD = st.secrets["DB_PASSWORD"]
    if "DB_NAME" in st.secrets:
        settings.DB_NAME = st.secrets["DB_NAME"]
    if "DB_PORT" in st.secrets:
        settings.DB_PORT = int(st.secrets["DB_PORT"])
except Exception:
    pass

if "db_engine" not in st.session_state or "db_agent" not in st.session_state:
    host = settings.DB_HOST
    port = settings.DB_PORT
    user = settings.DB_USER
    password = settings.DB_PASSWORD
    database = settings.DB_NAME
    
    engine = None
    db_agent = None
    if database:
        try:
            uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
            engine = create_engine(uri, connect_args={"connect_timeout": 5})
            with engine.connect() as conn:
                pass
            db_agent = DBAgent(host=host, port=port, user=user, password=password, database=database)
        except Exception as e:
            engine = None
            db_agent = DBAgent(host=host, port=port, user=user, password=password, database=database)
            db_agent.connection_error = f"Database Connection Error: {str(e)}"
    else:
        db_agent = DBAgent(host=host, port=port, user=user, password=password, database=database)
        db_agent.connection_error = "Database name is not configured."
        
    st.session_state.db_engine = engine
    st.session_state.db_agent = db_agent

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

# Helper to reload connection settings
def reconnect_db():
    host = st.session_state.db_host
    port = int(st.session_state.db_port)
    user = st.session_state.db_user
    password = st.session_state.db_pass
    database = st.session_state.db_name
    
    try:
        # Create standard SQLAlchemy connection engine
        uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        engine = create_engine(uri, connect_args={"connect_timeout": 5})
        # Test connection
        with engine.connect() as conn:
            pass
        
        # Instantiate DBAgent
        db_agent = DBAgent(host=host, port=port, user=user, password=password, database=database)
        
        st.session_state.db_engine = engine
        st.session_state.db_agent = db_agent
        
        # Update settings so they persist
        settings.DB_HOST = host
        settings.DB_PORT = port
        settings.DB_USER = user
        settings.DB_PASSWORD = password
        settings.DB_NAME = database
        
        st.toast("Successfully connected to MySQL database!", icon="✅")
    except Exception as e:
        db_agent = DBAgent(host=host, port=port, user=user, password=password, database=database)
        db_agent.connection_error = f"Database Connection Error: {str(e)}"
        st.session_state.db_engine = None
        st.session_state.db_agent = db_agent
        st.error(f"Failed to connect to database: {str(e)}")

# Helper to reload API keys
def update_llm_config():
    settings.LLM_PROVIDER = st.session_state.llm_provider
    settings.ANTHROPIC_API_KEY = st.session_state.anthropic_key
    settings.OPENAI_API_KEY = st.session_state.openai_key
    if "groq_key" in st.session_state:
        settings.GROQ_API_KEY = st.session_state.groq_key
    st.toast("LLM provider configuration updated!", icon="⚙️")


# --- SIDEBAR: Settings & Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/database.png", width=64)
    st.markdown("## Configuration Panel")
    
    # 1. API Configuration
    with st.expander("🔑 LLM Settings", expanded=True):
        provider_options = ["anthropic", "openai", "groq"]
        try:
            default_index = provider_options.index(settings.LLM_PROVIDER)
        except ValueError:
            default_index = 0

        st.selectbox(
            "LLM Provider",
            options=provider_options,
            index=default_index,
            key="llm_provider",
            on_change=update_llm_config
        )
        st.text_input(
            "Anthropic API Key",
            type="password",
            value=settings.ANTHROPIC_API_KEY or "",
            key="anthropic_key",
            on_change=update_llm_config
        )
        st.text_input(
            "OpenAI API Key",
            type="password",
            value=settings.OPENAI_API_KEY or "",
            key="openai_key",
            on_change=update_llm_config
        )
        if st.session_state.get("llm_provider", settings.LLM_PROVIDER) == "groq":
            st.text_input(
                "Enter Groq API Key",
                type="password",
                value=getattr(settings, "GROQ_API_KEY", "") or "",
                key="groq_key",
                on_change=update_llm_config
            )
        
    # 2. Database connection configuration
    with st.expander("🗄️ MySQL Connection", expanded=True):
        st.text_input("Host", value=settings.DB_HOST, key="db_host")
        st.number_input("Port", value=settings.DB_PORT, step=1, key="db_port")
        st.text_input("User", value=settings.DB_USER, key="db_user")
        st.text_input("Password", type="password", value=settings.DB_PASSWORD, key="db_pass")
        st.text_input("Database Name", value=settings.DB_NAME, key="db_name")
        st.button("Save & Connect", on_click=reconnect_db, use_container_width=True)

    # 3. CSV Dataset Uploader & Data Injector
    with st.expander("📤 Upload Custom Sample Dataset (CSV)", expanded=True):
        uploaded_csv = st.file_uploader("Choose a CSV file", type=["csv"])
        table_name = st.text_input("Target Table Name", placeholder="e.g. sales_data")
        
        if st.button("Inject Data into Database", use_container_width=True):
            if not st.session_state.get("db_engine"):
                st.error("No active database connection. Please connect to a database first.")
            elif not uploaded_csv:
                st.error("Please upload a CSV file.")
            elif not table_name.strip():
                st.error("Please enter a target table name.")
            else:
                try:
                    # Read the CSV with Pandas
                    df = pd.read_csv(uploaded_csv)
                    
                    # Write schema and data to database
                    db_engine = st.session_state.db_engine
                    table_name = table_name.strip()
                    
                    from sqlalchemy import BigInteger
                    from sqlalchemy.ext.compiler import compiles

                    # This explicitly defines a primary key type that SQLAlchemy understands
                    class MySQLPrimaryKeyBigInt(BigInteger):
                        pass

                    @compiles(MySQLPrimaryKeyBigInt, 'mysql')
                    def compile_pk_bigint(element, compiler, **kw):
                        return "BIGINT PRIMARY KEY"

                    # 1. Prepare the dataframe index
                    df.index = df.index + 1
                    df.index.name = 'id'

                    # 2. Force pandas to define 'id' as a PRIMARY KEY during initialization
                    df.to_sql(
                        name=table_name,
                        con=db_engine,
                        if_exists='replace',
                        index=True,
                        dtype={'id': MySQLPrimaryKeyBigInt()}
                    )
                    
                    # Re-instantiate DBAgent to immediately detect new table structure
                    active_agent = st.session_state.db_agent
                    st.session_state.db_agent = DBAgent(
                        host=active_agent.host,
                        port=active_agent.port,
                        user=active_agent.user,
                        password=active_agent.password,
                        database=active_agent.database
                    )
                    st.success(f"Successfully injected {len(df)} rows into table `{table_name.strip()}`!")
                    st.toast(f"Table `{table_name.strip()}` successfully created and indexed!", icon="🔄")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to inject data: {str(e)}")

    # 4. Document Uploader (RAG)
    with st.expander("📁 Document Indexer (RAG)", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload manuals or policy docs (.pdf, .txt)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily to docs directory
                temp_path = os.path.join(settings.DOCS_DIR, uploaded_file.name)
                
                # Check if it was already processed to avoid redundant builds
                if not os.path.exists(temp_path):
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner(f"Indexing {uploaded_file.name}..."):
                        success = st.session_state.rag_pipeline.ingest_document(temp_path)
                        if success:
                            st.toast(f"Successfully indexed {uploaded_file.name}!", icon="✅")
                        else:
                            st.error(f"Failed to index {uploaded_file.name}")
        
        # List indexed files
        files = os.listdir(settings.DOCS_DIR)
        if files:
            st.markdown("**Indexed Files:**")
            for f in files:
                st.markdown(f"- 📄 `{f}`")
        else:
            st.info("No documents indexed yet.")

    # 5. Clear chat
    if st.button("🗑️ Clear Conversational History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- MAIN PANEL: UI View ---
st.markdown("# 📊 Enterprise Business Intelligence Chatbot")
st.markdown("#### Natural Language to SQL, Document RAG, and Automatic Analytical Charting")

# Connection Health Badge Dashboard
col_status1, col_status2 = st.columns(2)
db_agent = st.session_state.db_agent

with col_status1:
    if db_agent.db and not db_agent.connection_error:
        # Fetch list of active tables
        try:
            from sqlalchemy import inspect
            tables = inspect(db_agent.db._engine).get_table_names()
            table_list_str = ", ".join(tables) if tables else "None"
            st.markdown(
                f'<div class="status-card"><b>MySQL Status:</b> <span class="badge-connected">Connected</span><br>'
                f'<small><b>Tables found:</b> {table_list_str}</small></div>',
                unsafe_allow_html=True
            )
        except Exception:
            st.markdown(
                '<div class="status-card"><b>MySQL Status:</b> <span class="badge-connected">Connected</span></div>',
                unsafe_allow_html=True
            )
    else:
        err_msg = db_agent.connection_error or "Configure connection parameters in the sidebar."
        st.markdown(
            f'<div class="status-card"><b>MySQL Status:</b> <span class="badge-disconnected">Disconnected</span><br>'
            f'<small style="color:#9B1C1C;">{err_msg}</small></div>',
            unsafe_allow_html=True
        )

with col_status2:
    indexed_files = os.listdir(settings.DOCS_DIR)
    has_index = os.path.exists(os.path.join(settings.VECTOR_DB_DIR, "index.faiss"))
    
    if has_index and indexed_files:
        st.markdown(
            f'<div class="status-card"><b>FAISS RAG Store:</b> <span class="badge-connected">Ready</span><br>'
            f'<small><b>Document count:</b> {len(indexed_files)} files</small></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-card"><b>FAISS RAG Store:</b> <span class="badge-disconnected">Empty</span><br>'
            '<small>Upload documentation in the sidebar to run text-based queries.</small></div>',
            unsafe_allow_html=True
        )


# --- Render Conversational History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render Route Indicator if present
        if "route" in msg:
            st.caption(f"🧭 Routed to: **{msg['route'].upper()}** | {msg.get('reasoning', '')}")
            
        # Render citations for RAG / Hybrid
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Document Source Citations"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"**Source {idx+1}:** `{src['source']}` (Page {src['page']})")
                    st.caption(src["content"])
                    st.markdown("---")

        # Render SQL block and Plotly express figures
        if "sql" in msg and msg["sql"]:
            with st.expander("💻 View Generated SQL Query"):
                st.code(msg["sql"], language="sql")
                
        if "dataframe" in msg and msg["dataframe"] is not None:
            df = pd.read_json(msg["dataframe"])
            st.markdown("**Query Results:**")
            st.dataframe(df, use_container_width=True)
            
            # Render chart if present
            chart_fig, chart_type = generate_auto_chart(df)
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
                st.caption(f"📈 Chart auto-generated based on result shape: **{chart_type.upper()}**")


# --- Process Chat Inputs ---
if user_query := st.chat_input("Ask a question about database metrics or documents..."):
    
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Process query response through Pipeline
    with st.chat_message("assistant"):
        response_container = st.empty()
        
        try:
            # Step A: Clean and Sanitise input
            cleaned_input = clean_natural_language_input(user_query)
            
            # Step B: Route Intent
            with st.spinner("Classifying intent..."):
                routing = route_query(cleaned_input)
                
            st.caption(f"🧭 Routed to: **{routing.route.upper()}** | {routing.reasoning}")
            
            # Prepare result dictionary
            assistant_response = {
                "role": "assistant",
                "route": routing.route,
                "reasoning": routing.reasoning,
                "content": "",
                "sql": "",
                "dataframe": None,
                "sources": []
            }
            
            # Step C: Execute target logic path
            if routing.route == "sql":
                with st.spinner("SQL Agent searching database..."):
                    sql_res = db_agent.ask_agent(cleaned_input)
                    
                assistant_response["content"] = sql_res["answer"]
                assistant_response["sql"] = sql_res["query"]
                
                if sql_res["dataframe"] is not None:
                    # Convert to JSON for streamlit session state storage
                    assistant_response["dataframe"] = sql_res["dataframe"].to_json()
                    
            elif routing.route == "rag":
                with st.spinner("FAISS searching indexed documents..."):
                    rag_res = st.session_state.rag_pipeline.query(cleaned_input)
                    
                assistant_response["content"] = rag_res["answer"]
                assistant_response["sources"] = rag_res["source_documents"]
                
            elif routing.route == "hybrid":
                # Run RAG
                with st.spinner("Retrieving document insights (1/2)..."):
                    rag_res = st.session_state.rag_pipeline.query(cleaned_input)
                
                # Run SQL Database Agent
                with st.spinner("Querying structural metrics (2/2)..."):
                    sql_res = db_agent.ask_agent(cleaned_input)
                
                # Synthesise response
                with st.spinner("Synthesising hybrid insight..."):
                    llm = get_llm(temperature=0.0)
                    
                    synthesis_system = (
                        "You are an expert Enterprise BI Assistant.\n"
                        "Your job is to merge findings from a relational database and textual documents "
                        "into a coherent, structured, and comprehensive answer.\n"
                        "State database values clearly and reference document insights explicitly.\n"
                        "Ensure the final response reads as one cohesive analysis."
                    )
                    
                    synthesis_body = (
                        f"USER QUESTION: {cleaned_input}\n\n"
                        f"DATABASE SUMMARY & METRICS:\n{sql_res['answer']}\n\n"
                        f"DOCUMENT DATA & DETAILS:\n{rag_res['answer']}"
                    )
                    
                    messages = [
                        ("system", synthesis_system),
                        ("user", synthesis_body)
                    ]
                    
                    synth_answer = llm.invoke(messages).content
                    
                assistant_response["content"] = synth_answer
                assistant_response["sql"] = sql_res["query"]
                assistant_response["sources"] = rag_res["source_documents"]
                
                if sql_res["dataframe"] is not None:
                    assistant_response["dataframe"] = sql_res["dataframe"].to_json()

            # Render assistant text response
            response_container.markdown(assistant_response["content"])
            
            # Show sources if any
            if assistant_response["sources"]:
                with st.expander("📚 View Document Source Citations"):
                    for idx, src in enumerate(assistant_response["sources"]):
                        st.markdown(f"**Source {idx+1}:** `{src['source']}` (Page {src['page']})")
                        st.caption(src["content"])
                        st.markdown("---")

            # Show SQL query if any
            if assistant_response["sql"]:
                with st.expander("💻 View Generated SQL Query"):
                    st.code(assistant_response["sql"], language="sql")
            
            # Show DataFrame and Chart if any
            if assistant_response["dataframe"] is not None:
                df = pd.read_json(assistant_response["dataframe"])
                st.markdown("**Query Results:**")
                st.dataframe(df, use_container_width=True)
                
                chart_fig, chart_type = generate_auto_chart(df)
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True)
                    st.caption(f"📈 Chart auto-generated based on result shape: **{chart_type.upper()}**")
            
            # Append message to conversation history
            st.session_state.messages.append(assistant_response)

        except SQLSecurityException as se:
            err_msg = f"⚠️ **Security Alert**: {str(se)}"
            response_container.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            
        except PromptInjectionException as pe:
            err_msg = f"⚠️ **Security Alert**: {str(pe)}"
            response_container.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            
        except Exception as e:
            logger.exception("General assistant processing error")
            err_msg = f"⚠️ **Application Error**: {str(e)}"
            response_container.markdown(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
