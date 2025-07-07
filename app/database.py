import os
import psycopg2
from fastapi import HTTPException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "mi_base_de_datos"),
    "user": os.getenv("DB_USER", "mi_usuario"),
    "password": os.getenv("DB_PASSWORD", "mi_contraseña_segura")
}

def get_db_connection():
    """Helper para obtener una conexión a la DB."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error al conectar a la base de datos: {e}")
        raise HTTPException(status_code=500, detail=f"No se pudo conectar a la base de datos: {e}")

def load_known_faces_from_db():
    """Carga nombres y embeddings (no nulos) desde la base de datos para reconocimiento."""
    known_face_encodings = []
    known_face_names = []
    
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT nombre, embedding FROM mis_personas WHERE embedding IS NOT NULL")
        rows = cur.fetchall()
        
        for nombre, embedding_data in rows:
            known_face_names.append(nombre)
            known_face_encodings.append(embedding_data)
        
        print(f"[{len(known_face_names)}] rostros (embeddings) cargados de la base de datos.")
        if known_face_names:
            print(f"Personas cargadas: {', '.join(known_face_names)}")

    except psycopg2.Error as e:
        print(f"Error al cargar rostros/embeddings de la DB: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la base de datos al cargar rostros: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()
    
    return known_face_encodings, known_face_names

def delete_face_from_db(nombre: str):
    """Elimina una persona de la base de datos por su nombre."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM mis_personas WHERE nombre = %s", (nombre,))
        conn.commit()
        return cur.rowcount > 0
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print(f"Error en la base de datos al eliminar: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la base de datos al eliminar: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

def save_face_to_db(nombre: str, embedding):
    """Guarda un nuevo rostro o actualiza uno existente en la base de datos."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Convertir el embedding a una lista para almacenamiento
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        
        # Verificar si ya existe un registro con el mismo nombre
        cur.execute("SELECT 1 FROM mis_personas WHERE nombre = %s", (nombre,))
        if cur.fetchone():
            # Actualizar el registro existente
            cur.execute(
                "UPDATE mis_personas SET embedding = %s WHERE nombre = %s",
                (embedding_list, nombre)
            )
        else:
            # Insertar un nuevo registro
            cur.execute(
                "INSERT INTO mis_personas (nombre, embedding) VALUES (%s, %s)",
                (nombre, embedding_list)
            )
        conn.commit()
        return True
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print(f"Error al guardar en la base de datos: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar en la base de datos: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

def get_db_status():
    """Verifica el estado de la conexión a la base de datos."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        return "Conectado"
    except Exception as e:
        return f"Error de conexión: {str(e)}"
    finally:
        if 'conn' in locals():
            cur.close()
            conn.close()
