"""Application configuration for MLR Profit Prediction."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    raw_data_path: str = "data/raw/50_Startups.csv"
    model_path: str = "app/models/mlr_profit_model.joblib"
    test_size: float = 0.2
    random_state: int = 0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
