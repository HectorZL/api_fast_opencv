from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from pydantic import BaseModel
import face_recognition
import numpy as np
import psycopg2
import os
import io
import cv2 # For image processing

#--ruta a la carpeta de las fotos--
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "reference_photos")

# --- Configuración de la Base de Datos ---
DB_CONFIG = {
    "host": "localhost",
    "database": "mi_base_de_datos", # ¡Asegúrate de que este sea el nombre correcto de tu DB!
    "user": "mi_usuario",
    "password": "mi_contraseña_segura"
}

def get_db_connection():
    """Helper para obtener una conexión a la DB."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error al conectar a la base de datos: {e}")
        raise HTTPException(status_code=500, detail=f"No se pudo conectar a la base de datos: {e}")

# --- Almacenamiento en memoria de los rostros conocidos ---
known_face_encodings = []
known_face_names = []
# Tolerancia para la comparación de rostros (ajustar según sea necesario, 0.6 es común)
FACE_RECOGNITION_TOLERANCE = 0.6
async def _register_face(nombre: str, local_image_path: str):
    """
    Lógica “pura” de registro, sin depender de Form (…) de FastAPI.
    Recibe nombre y ruta local, procesa la imagen y hace INSERT/UPDATE.
    """
    # 1. Verificar existencia de archivo
    if not os.path.exists(local_image_path) or not os.path.isfile(local_image_path):
        raise HTTPException(status_code=400, detail=f"La ruta de imagen '{local_image_path}' no es válida.")
    # 2. Leer imagen, extraer embedding…
    with open(local_image_path, 'rb') as f:
        foto_bytes = f.read()
    nparr = np.frombuffer(foto_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb)
    new_embedding_list = None
    if locs:
        emb = face_recognition.face_encodings(rgb, locs)[0]
        new_embedding_list = emb.tolist()
    # 3. Conectar a BD y realizar INSERT o UPDATE (igual que antes)…
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT 1 FROM mis_personas WHERE nombre = %s", (nombre,))
    if cur.fetchone():
        # Actualiza el embedding existente
        cur.execute(
            "UPDATE mis_personas SET embedding = %s WHERE nombre = %s",
            (new_embedding_list, nombre)
        )
    else:
        # Inserta un nuevo registro
        cur.execute(
            "INSERT INTO mis_personas (nombre, embedding) VALUES (%s, %s)",
            (nombre, new_embedding_list)
        )
    conn.commit()
    cur.close()
    conn.close()
    # 4. Recargar memoria
    load_known_faces_from_db()
    return {"message": f"'{nombre}' registrado/actualizado.", "nombre": nombre}

def load_known_faces_from_db():
    """
    Carga nombres y embeddings (no nulos) desde la base de datos para reconocimiento.
    """
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Seleccionamos solo las entradas que tienen un embedding (no es NULL)
        cur.execute("SELECT nombre, embedding FROM mis_personas WHERE embedding IS NOT NULL")
        rows = cur.fetchall()
        
        for nombre, embedding_data in rows:
            # Convierte la lista de Python (que viene de REAL[128]) a un array NumPy
            known_face_names.append(nombre)
            known_face_encodings.append(np.array(embedding_data)) 
        
        print(f"[{len(known_face_names)}] rostros (embeddings) cargados de la base de datos para la API.")
        if known_face_names:
            print(f"Personas cargadas: {', '.join(known_face_names)}")

    except HTTPException: # Re-lanzar si la conexión falló en get_db_connection
        raise
    except psycopg2.Error as e:
        print(f"Error al cargar rostros/embeddings de la DB: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la base de datos al cargar rostros: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

# --- Modelo Pydantic para la respuesta de reconocimiento ---
class RecognitionResult(BaseModel):
    name: str
    is_known: bool
    distance: float = None # Distancia euclidiana al rostro más cercano

# --- Inicializar FastAPI ---
app = FastAPI(
    title="API de Reconocimiento Facial (PostgreSQL con Embeddings y Anti-Duplicado)",
    description="API para registrar y reconocer rostros usando PostgreSQL, con verificación de duplicados."
)

@app.on_event("startup")
async def startup_event():
    """Se ejecuta cuando la aplicación FastAPI arranca."""
    print("Iniciando API de Reconocimiento Facial...")
    load_known_faces_from_db() # Cargar los rostros conocidos desde la base de datos.
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            nombre = os.path.splitext(filename)[0]
            local_image_path = os.path.join(IMAGE_FOLDER, filename)
            try:
                await _register_face(nombre, local_image_path)
                print(f"Registrado automáticamente: {nombre}")
            except Exception as e:
                print(f"Error al registrar {nombre}: {e}")
                
@app.post("/register_face_from_local_path/")
async def register_face_from_local_path(
    nombre: str = Form(...),
    local_image_path: str = Form(...)
):
    return await _register_face(nombre, local_image_path)


@app.post("/recognize_face/")
async def recognize_face(file: UploadFile = File(...)):
    """
    Compara una nueva foto subida con los rostros cargados de la base de datos.
    Retorna el nombre del rostro reconocido o "Desconocido".
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Formato de archivo no válido. Se esperaba una imagen.")

    # Leer la imagen desde el UploadFile
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen subida.")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    if not face_locations:
        return RecognitionResult(name="No Rostro Detectado", is_known=False, distance=None)

    # Asumimos que procesamos el primer rostro detectado para simplificar
    test_face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]

    if not known_face_encodings:
        return RecognitionResult(name="No hay rostros de referencia cargados en memoria.", is_known=False, distance=None)

    # Realizar la comparación (lógica de face_recognition)
    matches = face_recognition.compare_faces(known_face_encodings, test_face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
    face_distances = face_recognition.face_distance(known_face_encodings, test_face_encoding)

    best_match_index = np.argmin(face_distances)
    
    if matches[best_match_index]:
        name = known_face_names[best_match_index]
        return RecognitionResult(name=name, is_known=True, distance=float(face_distances[best_match_index]))
    else:
        return RecognitionResult(name="Desconocido", is_known=False, distance=float(face_distances[best_match_index]))

@app.delete("/delete_face/{nombre}")
async def delete_face(nombre: str):
    """Elimina una persona registrada de la base de datos por su nombre."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM mis_personas WHERE nombre = %s", (nombre,))
        conn.commit()
        if cur.rowcount > 0:
            load_known_faces_from_db() # Recargar los embeddings en memoria de la API
            return {"message": f"Persona '{nombre}' eliminada exitosamente de la DB."}
        else:
            raise HTTPException(status_code=404, detail=f"Persona con nombre '{nombre}' no encontrada.")
    except HTTPException:
        raise
    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error en la base de datos al eliminar: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la base de datos al eliminar: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

@app.get("/list_known_faces/")
async def list_known_faces():
    """Lista todos los nombres de las personas con embeddings cargados."""
    return {"known_faces": known_face_names}

@app.get("/status/")
async def get_status():
    """Retorna el estado de la API y el número de rostros cargados."""
    db_status = "Desconectado"
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        db_status = "Conectado"
    except HTTPException as e:
        db_status = f"Error de conexión: {e.detail}"
    except psycopg2.Error as e:
        db_status = f"Error de DB: {e}"
    finally:
        if conn:
            conn.close()

    return {
        "status": "running",
        "loaded_faces_count": len(known_face_names),
        "source": "Cargado desde la tabla 'mis_personas' de PostgreSQL",
        "database_connection": db_status,
        "message": "Listo para reconocer rostros."
    }

@app.post("/refresh_db_faces/")
async def refresh_db_faces():
    """Recarga los rostros de referencia desde la base de datos."""
    load_known_faces_from_db()
    return {"message": f"Rostros recargados desde la DB. Total: {len(known_face_names)}"}