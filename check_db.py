"""Check that DATABASE_URL in .env can reach Postgres."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str


def main() -> None:
    settings = Settings()
    url = make_url(settings.database_url)
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar_one()
    print(f"connected to {url.host}:{url.port}/{url.database}")
    print(version)


if __name__ == "__main__":
    main()
