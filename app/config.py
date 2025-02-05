from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str
    JWT_SECRET: str
    DATABASE_URL: str
    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    UPLOAD_DIR: str = "uploads"

    class Config:
        env_file = ".env"


settings = Settings()
