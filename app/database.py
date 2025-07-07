import os
import psycopg2
import base64
import json
import numpy as np
from fastapi import HTTPException
from dotenv import load_dotenv
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Load environment variables
load_dotenv()

# Database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "mi_base_de_datos"),
    "user": os.getenv("DB_USER", "mi_usuario"),
    "password": os.getenv("DB_PASSWORD", "mi_contraseña_segura")
}

# Get secret key from environment and derive a Fernet key
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-please-change-me")
salt = b'salt_'  # You might want to store this more securely in production

# Derive a 32-byte key from the secret key using PBKDF2
def get_fernet_key():
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(SECRET_KEY.encode()))

# Initialize Fernet with our derived key
fernet = Fernet(get_fernet_key())

def encrypt_data(data):
    """Encrypt data using Fernet symmetric encryption."""
    # Convert numpy array to JSON string if it's a numpy array
    if isinstance(data, np.ndarray):
        data = json.dumps(data.tolist())
    # If it's already a string, ensure it's bytes
    if isinstance(data, str):
        data = data.encode()
    return fernet.encrypt(data).decode()

def decrypt_data(encrypted_data):
    """Decrypt data using Fernet symmetric encryption."""
    if not encrypted_data:
        return None
    try:
        decrypted = fernet.decrypt(encrypted_data.encode() if isinstance(encrypted_data, str) else encrypted_data)
        # Try to convert back to numpy array if it was a JSON array
        try:
            return np.array(json.loads(decrypted.decode()))
        except (json.JSONDecodeError, ValueError):
            return decrypted.decode()
    except Exception as e:
        print(f"Decryption error: {e}")
        return None

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
        # Seleccionamos solo las entradas que tienen un encoding_hash (no es NULL)
        cur.execute("SELECT nombre, encoding_hash FROM mis_personas WHERE encoding_hash IS NOT NULL")
        rows = cur.fetchall()
        
        for nombre, encrypted_embedding in rows:
            # Desencriptar el embedding
            embedding_data = decrypt_data(encrypted_embedding)
            if embedding_data is not None:
                known_face_names.append(nombre)
                known_face_encodings.append(embedding_data)
        
        print(f"[{len(known_face_names)}] rostros (embeddings) cargados de la base de datos.")
        if known_face_names:
            print(f"Personas cargadas: {', '.join(known_face_names)}")

    except HTTPException: # Re-lanzar si la conexión falló en get_db_connection
        raise
    except psycopg2.Error as e:
        print(f"Error al cargar rostros/embeddings de la DB: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la base de datos al cargar rostros: {e}")
    except Exception as e:
        print(f"Error inesperado al cargar rostros: {e}")
        raise HTTPException(status_code=500, detail=f"Error inesperado al cargar rostros: {e}")
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
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error inesperado al eliminar de la base de datos: {e}")
        raise HTTPException(status_code=500, detail=f"Error inesperado al eliminar: {e}")
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
        
        # Encriptar el embedding antes de guardarlo
        encrypted_embedding = encrypt_data(embedding)
        
        # Verificar si ya existe un registro con el mismo nombre
        cur.execute("SELECT 1 FROM mis_personas WHERE nombre = %s", (nombre,))
        if cur.fetchone():
            # Actualizar el registro existente
            cur.execute(
                "UPDATE mis_personas SET encoding_hash = %s WHERE nombre = %s",
                (encrypted_embedding, nombre)
            )
        else:
            # Insertar un nuevo registro
            cur.execute(
                "INSERT INTO mis_personas (nombre, encoding_hash) VALUES (%s, %s)",
                (nombre, encrypted_embedding)
            )
        conn.commit()
        return True
    except psycopg2.Error as e:
        if conn:
            conn.rollback()
        print(f"Error al guardar en la base de datos: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar en la base de datos: {e}")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error inesperado al guardar en la base de datos: {e}")
        raise HTTPException(status_code=500, detail=f"Error inesperado al guardar en la base de datos: {e}")
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
