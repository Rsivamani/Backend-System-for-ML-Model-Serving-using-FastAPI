"""Application configuration for DT Classification."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    raw_data_path: str = "data/raw/Social_Network_Ads.csv"
    model_path: str = "app/models/dt_clf_model.joblib"
    test_size: float = 0.3
    random_state: int = 0
    max_depth: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
