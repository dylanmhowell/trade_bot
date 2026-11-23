from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # TopstepX / ProjectX Gateway API
    # Auth: POST /api/Auth/loginKey with { userName, apiKey }
    topstep_username: str = ""
    topstep_api_key: str = ""
    topstep_base_url: str = "https://api.topstepx.com"
    topstep_account_id: int | None = None

    # Market data / contract selection
    # Practice/sim accounts usually need live=false on Contract/search
    topstep_contract_id: str = ""
    topstep_symbol_search: str = "MNQ"
    topstep_contract_live: bool = False
    topstep_rtc_url: str = "https://rtc.topstepx.com/hubs/market"

    # When true, live-mode strategy actions place real broker orders
    # (still only against whatever account Topstep returns — often simulated)
    topstep_enable_orders: bool = True

    # Strategy parameters (can also come from env)
    stop_ticks: int = 240
    lots: int = 2
    tp1_ticks: int = 80


settings = Settings()
