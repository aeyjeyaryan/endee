from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Endee
    endee_base_url: str = Field("http://localhost:8080/api/v1", alias="ENDEE_BASE_URL")
    endee_auth_token: str = Field("", alias="ENDEE_AUTH_TOKEN")
    endee_index_name: str = Field("docurag_index", alias="ENDEE_INDEX_NAME")

    # Embeddings
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # Chunking
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")

    # LLM
    llm_provider: str = Field("gemini", alias="LLM_PROVIDER")          # "gemini" | "ollama"
    gemini_api_key: str = Field("", alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-1.5-flash", alias="GEMINI_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3", alias="OLLAMA_MODEL")

    # Retrieval
    top_k: int = Field(5, alias="TOP_K")

    # API
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")


# Singleton settings object — import this everywhere
settings = Settings()