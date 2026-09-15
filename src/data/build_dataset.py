import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config.settings import Settings

def load_dataset():
    settings = Settings.from_yaml()
    conn = psycopg2.connect(
        host=settings.env.db_host,
        port=settings.env.db_port,
        user=settings.env.db_user,
        password=settings.env.db_password,
        dbname=settings.env.db_name
    )
    query = "SELECT * FROM transactions"
    df = pd.read_sql(query, conn, index_col="id")
    conn.close()
    return df

def build_dataset():
    df = load_dataset()
    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test