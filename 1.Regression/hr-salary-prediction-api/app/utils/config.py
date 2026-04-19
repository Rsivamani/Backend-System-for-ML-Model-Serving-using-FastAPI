"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Data
    raw_data_path: str = "data/raw/Salary_Data.csv"

    # Model artefact
    model_path: str = "app/models/slr_salary_model.joblib"

    # Training hyperparameters
    test_size: float = 0.3
    random_state: int = 0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
