"""Application configuration for Decision Tree Regression."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    raw_data_path: str = "data/raw/50_Startups.csv"
    model_path: str = "app/models/dtr_profit_model.joblib"
    test_size: float = 0.2
    random_state: int = 0
    max_depth: int = 5
    min_samples_split: int = 2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
