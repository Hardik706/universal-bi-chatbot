import os
from typing import Any
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
from langchain_groq import ChatGroq

# Load environment variables from a .env file if it exists
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    """
    # LLM Settings
    LLM_PROVIDER: str = Field(default="anthropic", description="LLM provider: 'anthropic', 'openai', or 'groq'")
    ANTHROPIC_API_KEY: str | None = Field(default=None, description="Anthropic API Key")
    OPENAI_API_KEY: str | None = Field(default=None, description="OpenAI API Key")
    GROQ_API_KEY: str | None = Field(default=None, description="Groq API Key")
    
    # Model selections
    ANTHROPIC_MODEL: str = Field(default="claude-3-5-sonnet-20241022", description="Anthropic model name")
    OPENAI_MODEL: str = Field(default="llama-3.3-70b-versatile", description="OpenAI/Groq model name")
    TEMPERATURE: float = Field(default=0.0, description="Temperature for the LLM")

    # MySQL Database Settings
    DB_HOST: str = Field(default="localhost", description="MySQL Database Host")
    DB_PORT: int = Field(default=3306, description="MySQL Database Port")
    DB_USER: str = Field(default="root", description="MySQL Database User Name")
    DB_PASSWORD: str = Field(default="", description="MySQL Database Password")
    DB_NAME: str = Field(default="", description="MySQL Database Name")

    # Vector DB / FAISS settings
    VECTOR_DB_DIR: str = Field(default=str(BASE_DIR / "data" / "faiss_index"), description="Directory to store FAISS index")
    DOCS_DIR: str = Field(default=str(BASE_DIR / "docs"), description="Directory for raw documents to upload/read")

    @computed_field
    @property
    def database_uri(self) -> str:
        """
        Computes the SQLAlchemy database URI for MySQL connection.
        """
        # Return empty/mock connection string if DB details are missing to prevent runtime failures before user configures it
        if not self.DB_NAME:
            return "mysql+pymysql://root@localhost/temp_db"
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings singleton
settings = Settings()

# Try loading from streamlit secrets if available
try:
    import streamlit as st
    if hasattr(st, "secrets"):
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
            
        if "LLM_PROVIDER" in st.secrets:
            settings.LLM_PROVIDER = st.secrets["LLM_PROVIDER"]
        if "ANTHROPIC_API_KEY" in st.secrets:
            settings.ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
        if "OPENAI_API_KEY" in st.secrets:
            settings.OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
        if "GROQ_API_KEY" in st.secrets:
            settings.GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# Ensure directories exist
os.makedirs(settings.DOCS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.VECTOR_DB_DIR), exist_ok=True)

def get_llm(temperature: float = 0.0) -> Any:
    """
    Utility factory to retrieve the configured Chat Model (Anthropic, OpenAI, or Groq).
    """
    from langchain_core.language_models import BaseChatModel
    import streamlit as st
    
    if settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # Try getting from settings first, then st.secrets, then env.
        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            try:
                if "ANTHROPIC_API_KEY" in st.secrets:
                    api_key = st.secrets["ANTHROPIC_API_KEY"]
            except Exception:
                pass
        if not api_key:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")

        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            temperature=temperature,
            api_key=api_key
        )
    elif settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            try:
                if "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
            except Exception:
                pass
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured.")

        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=temperature,
            api_key=api_key
        )
    elif settings.LLM_PROVIDER == "groq":
        api_key = settings.GROQ_API_KEY
        if not api_key:
            try:
                if "GROQ_API_KEY" in st.secrets:
                    api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

