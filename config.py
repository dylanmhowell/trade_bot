from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # TopstepX / ProjectX Gateway API
    # Auth: POST /api/Auth/loginKey with { userName, apiKey }
    topstep_username: str = ""
    topstep_api_key: str = ""
    topstep_base_url: str = "https://api.topstepx.com"
    topstep_account_id: int | None = None

    # Strategy parameters (can also come from env)
    stop_ticks: int = 240
    lots: int = 2
    tp1_ticks: int = 80


settings = Settings()
