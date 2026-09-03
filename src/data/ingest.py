import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_required_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Environment variable '{var_name}' is missing")
    return value

def download_dataset(dataset : str, output_dir: Path):
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset, path=str(output_dir), unzip=True)


def load_data(cursor, csv_path: Path):
    df = pd.read_csv(csv_path)
    columns = ','.join(df.columns)
    with open(csv_path, 'r') as f:
        cursor.copy_expert(
            f"COPY transactions ({columns}) FROM STDIN WITH CSV HEADER",
            f
        )

def main():
    load_dotenv()
    config = load_config()
    kaggle_api_token = get_required_env("KAGGLE_API_TOKEN")
    db_host = get_required_env("DB_HOST")
    db_port = int(get_required_env("DB_PORT"))
    db_user = get_required_env("DB_USER")
    db_password = get_required_env("DB_PASSWORD")
    db_name = get_required_env("DB_NAME")
    
    dataset = config["DATASET"]
    output_dir = config["OUTPUT_DIR"]
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    download_dataset(dataset, output_dir)

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        dbname=db_name
    )
    try:
        cursor = conn.cursor()
        load_data(cursor, output_dir / "creditcard.csv")
        conn.commit()
        print("Dataset loaded")
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()