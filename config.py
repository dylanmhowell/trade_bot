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
    # Champion: ATR5×0.75 TP1 + BE@TP1 + opp VA +25pt + no Friday
    stop_ticks: int = 240   # 60 points
    lots: int = 2
    tp1_ticks: int = 160    # fallback when ATR sizing off / not warm
    move_be: bool = True
    opp_pull_ticks: int = -100  # negative = push beyond opp VA (25pt)
    no_friday: bool = True
    atr_mult: float = 0.75
    atr_lookback_days: int = 5
    tp1_min_ticks: int = 60   # 15 points
    tp1_max_ticks: int = 400  # 100 points


settings = Settings()
