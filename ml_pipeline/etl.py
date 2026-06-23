import logging
import sqlite3
from pathlib import Path

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def persist_to_sqlite(df: pd.DataFrame, sqlite_path: Path, table_name: str = "transactions") -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Persisting cleaned transactions to SQLite database %s", sqlite_path)

    with sqlite3.connect(sqlite_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()

    logging.info("Persisted %s rows to SQLite table %s", len(df), table_name)
