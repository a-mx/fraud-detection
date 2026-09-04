import os
from dataclasses import dataclass
from dotenv import load_dotenv
@dataclass(frozen=True)
class EnvConfig:
    kaggle_api_token: str
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str

    @classmethod
    def from_env(cls) -> "EnvConfig":
        load_dotenv()
        def required(name: str) -> str:
            value = os.getenv(name)
            if value is None or value == "":
                raise ValueError(f"Environment variable '{name}' is missing")
            return value

        return cls(
            kaggle_api_token=required("KAGGLE_API_TOKEN"),
            db_host=required("DB_HOST"),
            db_port=int(required("DB_PORT")),
            db_user=required("DB_USER"),
            db_password=required("DB_PASSWORD"),
            db_name=required("DB_NAME"),
        )
    
if __name__ == "__main__":
    pass