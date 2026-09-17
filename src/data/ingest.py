import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import yaml
from src.config.settings import Settings


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
    settings = Settings.from_yaml()
    
    dataset = settings.dataset.name
    output_dir = settings.dataset.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dataset(dataset, output_dir)

    conn = psycopg2.connect(
        host=settings.env.db_host,
        port=settings.env.db_port,
        user=settings.env.db_user,
        password=settings.env.db_password,
        dbname=settings.env.db_name
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