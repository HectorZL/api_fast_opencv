import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from fastapi import HTTPException
import json
import numpy as np
from typing import List, Tuple, Optional
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'midbnew'),
    'user': os.getenv('DB_USER', 'mi_usuario'),
    'password': os.getenv('DB_PASSWORD', 'mi_contraseña'),
    'port': int(os.getenv('DB_PORT', 3306))
}

# Get secret key from environment and derive a Fernet key
SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key-please-change-me")
salt = b'salt_'  # You might want to store this more securely in production

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
    if isinstance(data, np.ndarray):
        data = json.dumps(data.tolist())
    if isinstance(data, str):
        data = data.encode()
    return fernet.encrypt(data).decode()

def decrypt_data(encrypted_data):
    """Decrypt data using Fernet symmetric encryption."""
    if not encrypted_data:
        return None
    try:
        decrypted = fernet.decrypt(encrypted_data.encode() if isinstance(encrypted_data, str) else encrypted_data)
        try:
            return np.array(json.loads(decrypted.decode()))
        except (json.JSONDecodeError, ValueError):
            return decrypted.decode()
    except Exception as e:
        print(f"Decryption error: {e}")
        return None

def get_db_connection():
    """Get a database connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error connecting to MySQL database: {e}"
        )

def save_face_embedding(nombre: str, embedding: List[float]) -> int:
    """Save or update a face embedding in the database."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # Convert the embedding list to a JSON string and then to bytes
            embedding_json = json.dumps(embedding)
            encrypted_embedding = encrypt_data(embedding_json)
        except Exception as e:
            print(f"Error encrypting data: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error encrypting face data: {str(e)}")
        
        # Check if person already exists
        cursor.execute(
            "SELECT id FROM personas WHERE nombre = %s AND estado = 1",
            (nombre,)
        )
        person = cursor.fetchone()
        
        if person:
            # Update existing record
            cursor.execute(
                """UPDATE personas 
                   SET codigo_embedding = %s, 
                       estado = 1,
                       fecha_actualizacion = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (encrypted_embedding, person['id'])
            )
            person_id = person['id']
        else:
            # Insert new record
            cursor.execute(
                """INSERT INTO personas 
                   (nombre, codigo_embedding, estado) 
                   VALUES (%s, %s, 1)""",
                (nombre, encrypted_embedding)
            )
            person_id = cursor.lastrowid
        
        conn.commit()
        return person_id
        
    except Error as e:
        if conn and conn.is_connected():
            conn.rollback()
        print(f"Database error in save_face_embedding: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        if conn and conn.is_connected():
            conn.rollback()
        print(f"Unexpected error in save_face_embedding: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def get_face_embeddings() -> List[Tuple[int, str, List[float]]]:
    """Retrieve all face embeddings from the database."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT id, nombre, codigo_embedding FROM personas WHERE estado = 1"
        )
        
        results = []
        for row in cursor:
            try:
                # Decrypt the embedding
                embedding = decrypt_data(row['codigo_embedding'])
                results.append((
                    row['id'],
                    row['nombre'],
                    embedding
                ))
            except (json.JSONDecodeError, KeyError):
                continue
                
        return results
        
    except Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving face embeddings: {e}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def delete_face(persona_id: int) -> bool:
    """Delete a face from the database (soft delete)."""
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE personas SET estado = 0 WHERE id = %s",
            (persona_id,)
        )
        conn.commit()
        
        return cursor.rowcount > 0
        
    except Error as e:
        if conn and conn.is_connected():
            conn.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting face: {e}"
        )
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

def load_known_faces_from_db():
    """Carga nombres y embeddings (no nulos) desde la base de datos para reconocimiento."""
    known_face_encodings = []
    known_face_names = []
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT nombre, codigo_embedding FROM personas WHERE estado = 1")
        rows = cursor.fetchall()
        
        for row in rows:
            nombre = row['nombre']
            encrypted_embedding = row['codigo_embedding']
            
            # Desencriptar el embedding
            try:
                embedding = decrypt_data(encrypted_embedding)
                if embedding is not None:
                    known_face_encodings.append(embedding)
                    known_face_names.append(nombre)
            except Exception as e:
                print(f"Error al desencriptar embedding para {nombre}: {e}")
        
        print(f"[{len(known_face_names)}] rostros (embeddings) cargados de la base de datos.")
        if known_face_names:
            print(f"Personas cargadas: {', '.join(known_face_names)}")

        return known_face_encodings, known_face_names
        
    except Error as e:
        print(f"Error al cargar rostros/embeddings de la DB: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la base de datos al cargar rostros: {e}")
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def get_db_status():
    """Verifica el estado de la conexión a la base de datos."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        return "Conectado"
    except Exception as e:
        return f"Error de conexión: {str(e)}"
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
