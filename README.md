# 📊 LLM-Powered Business Intelligence Chatbot

An enterprise-grade, production-ready AI agent that connects to structured MySQL databases and unstructured doc stores (RAG) to provide natural language insights, SQL generation, automatic Plotly charting, and semantic search.

---

## 🛠️ Technology Stack

- **Core Framework**: LangChain (v0.2+)
- **Interface**: Streamlit
- **Relational Database**: MySQL via SQLAlchemy (`mysql+mysqlconnector`)
- **Unstructured Search**: FAISS vector store
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Document Parser**: PyMuPDF (`fitz`)
- **Visualization**: Plotly Express

---

## 🏗️ Folder Structure

```
bi-chatbot/
├── app.py                  # Streamlit UI & Orchestration Flow
├── config.py               # Settings validation & LLM factory
├── requirements.txt        # Package dependencies
├── README.md               # Setup and usage guidelines
├── core/
│   ├── db_agent.py         # LangChain SQL Database Agent + security tools
│   ├── rag_chain.py        # PyMuPDF processing + FAISS index & query search
│   └── router.py           # Structured output classifier (SQL vs. RAG vs. Hybrid)
├── utils/
│   ├── chart_engine.py     # Auto Plotly Chart Router based on dataframe shapes
│   ├── schema_extractor.py # Dynamic SQLAlchemy schema reader
│   └── sanitiser.py        # Input sanitisation & SQL command safety guards
└── data/                   # Default storage for raw docs & FAISS indices
```

---

## ⚙️ Setup and Installation

### 1. Prerequisite Environments
Ensure you have **Python 3.10+** and a running **MySQL** instance.

### 2. Clone and Install Dependencies
Navigate to the project root directory and run:
```bash
pip install -r requirements.txt
```

### 3. Setup Configuration
Create a `.env` file in the `bi-chatbot/` directory:
```env
# LLM Settings
LLM_PROVIDER=anthropic             # Options: 'anthropic' or 'openai'
ANTHROPIC_API_KEY=your-anthropic-api-key
# OPENAI_API_KEY=your-openai-api-key

# Database Settings
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_database_name
```

---

## 🛡️ Core Features & Architectures

### 1. Query Routing Flow
Every prompt is evaluated by a structured intent classifier before execution:
```
                     ┌───────────────────┐
                     │    User Input     │
                     └─────────┬─────────┘
                               │
                       [ Sanitiser Check ]
                               │
                     ┌─────────▼─────────┐
                     │   Intent Router   │
                     └────┬─────┬─────┬──┘
                          │     │     │
                 ┌────────┘     │     └────────┐
                 ▼              ▼              ▼
            ┌─────────┐   ┌───────────┐   ┌─────────┐
            │   SQL   │   │  Hybrid   │   │   RAG   │
            └────┬────┘   └─────┬─────┘   └────┬────┘
                 │              │              │
                 │         [Run RAG &   │
                 │          SQL Agent]         │
                 ▼              ▼              ▼
           [SQL Agent]   [Synthesis LLM] [FAISS Search]
                 │              │              │
                 └──────────────┼──────────────┘
                                ▼
                       ┌─────────────────┐
                       │  Render Text +  │
                       │  Plotly Chart   │
                       └─────────────────┘
```

### 2. Database Agent & Security Boundary
- The SQL agent has dynamic schema awareness using [schema_extractor.py](file:///c:/Users/Hardik/Desktop/LLM_BT_Chatbot/bi-chatbot/utils/schema_extractor.py).
- **Execution Interception**: The default tool executing SQL is swapped out with `SecureQuerySQLDatabaseTool`. Any query that does not start with an allowed read-only word (`SELECT`, `SHOW`, `DESCRIBE`, `EXPLAIN`, `WITH`) or contains blocked write commands (`DROP`, `DELETE`, `ALTER`, `TRUNCATE`, `INSERT`, `UPDATE`, `REPLACE`) is immediately blocked.

### 3. Auto-Charting Logic
DataFrames returned from the SQL database are automatically plotted using [chart_engine.py](file:///c:/Users/Hardik/Desktop/LLM_BT_Chatbot/bi-chatbot/utils/chart_engine.py):
- **Line Chart (`px.line`)**: If dataset has 2 columns, where col 1 is date/month and col 2 is numeric.
- **Pie Chart (`px.pie`)**: If dataset has 2 columns and row count is $\le$ 8.
- **Bar Chart (`px.bar`)**: If dataset has 2 columns and row count is $>$ 8.
- **Table View (Fallback)**: For complex multi-column structures or non-numeric sets.

---

## 🚀 Running the Chatbot
Launch the Streamlit app:
```bash
streamlit run app.py
```
Use the sidebar to adjust DB settings, API keys, and upload manuals for instantaneous FAISS document embedding and querying.
