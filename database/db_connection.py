from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:password@localhost/frauddb"

engine = create_engine(DATABASE_URL)