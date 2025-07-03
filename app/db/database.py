import psycopg2
from fastapi import HTTPException
from ..config import settings

def get_db_connection():
    """Helper para obtener una conexión a la DB."""
    try:
        conn = psycopg2.connect(
            host=settings.DB_HOST,
            database=settings.DB_NAME,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error al conectar a la base de datos: {e}")
        raise HTTPException(status_code=500, detail=f"No se pudo conectar a la base de datos: {e}")
