from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from pydantic import BaseModel
import os
import uvicorn
import numpy as np
from typing import List, Optional
from pathlib import Path

# Import our modules
from .database import (
    get_db_connection, 
    load_known_faces_from_db, 
    delete_face_from_db,
    save_face_to_db,
    get_db_status
)
from .face_utils import (
    process_image,
    extract_face_embedding,
    recognize_face as recognize_face_util,
    process_and_register_face,
    FaceRecognitionResult
)

# --- Configuración de la aplicación ---
BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "reference_photos"

# Crear el directorio de imágenes si no existe
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# --- Modelos Pydantic ---
class RecognitionResult(BaseModel):
    name: str
    is_known: bool
    distance: Optional[float] = None

# --- Inicializar FastAPI ---
app = FastAPI(
    title="API de Reconocimiento Facial",
    description="API para registrar y reconocer rostros usando PostgreSQL",
    version="1.0.0"
)

# Almacenamiento en memoria de los rostros conocidos
known_face_encodings: List[np.ndarray] = []
known_face_names: List[str] = []

# --- Eventos de la aplicación ---
@app.on_event("startup")
async def startup_event():
    """Se ejecuta cuando la aplicación FastAPI arranca."""
    print("Iniciando API de Reconocimiento Facial...")
    
    # Cargar rostros conocidos desde la base de datos
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces_from_db()
    
    # Registrar automáticamente las imágenes en la carpeta reference_photos
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            nombre = os.path.splitext(filename)[0]
            local_image_path = os.path.join(IMAGE_FOLDER, filename)
            
            try:
                with open(local_image_path, 'rb') as f:
                    image_data = f.read()
                
                # Procesar y registrar el rostro
                result = process_and_register_face(
                    image_data=image_data,
                    nombre=nombre,
                    known_face_encodings=known_face_encodings,
                    known_face_names=known_face_names
                )
                
                # Guardar el embedding en la base de datos
                face_encoding = np.array(result['face_encoding'])
                save_face_to_db(nombre, face_encoding)
                
                print(f"Registrado automáticamente: {nombre}")
                
            except Exception as e:
                print(f"Error al registrar {nombre}: {e}")

# --- Endpoints de la API ---
@app.post("/register_face/")
async def register_face(
    file: UploadFile = File(...),
    nombre: str = Form(...)
):
    """
    Registra un nuevo rostro en el sistema.
    
    - **file**: Imagen que contiene el rostro a registrar
    - **nombre**: Nombre de la persona en la imagen
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Formato de archivo no válido. Se esperaba una imagen.")
    
    # Leer la imagen
    contents = await file.read()
    
    # Procesar y registrar el rostro
    result = process_and_register_face(
        image_data=contents,
        nombre=nombre,
        known_face_encodings=known_face_encodings,
        known_face_names=known_face_names
    )
    
    # Guardar el embedding en la base de datos
    face_encoding = np.array(result['face_encoding'])
    save_face_to_db(nombre, face_encoding)
    
    return result

@app.post("/recognize_face/")
async def recognize_face_endpoint(file: UploadFile = File(...)) -> RecognitionResult:
    """
    Compara una foto subida con los rostros conocidos.
    
    - **file**: Imagen que contiene el rostro a reconocer
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Formato de archivo no válido. Se esperaba una imagen.")
    
    # Leer y procesar la imagen
    contents = await file.read()
    rgb_image = process_image(contents)
    
    # Extraer el rostro
    success, face_encoding = extract_face_embedding(rgb_image)
    if not success or face_encoding is None:
        return RecognitionResult(name="No se detectó ningún rostro", is_known=False)
    
    # Reconocer el rostro
    result = recognize_face_util(
        test_face_encoding=face_encoding,
        known_face_encodings=known_face_encodings,
        known_face_names=known_face_names
    )
    
    return RecognitionResult(
        name=result.name,
        is_known=result.is_known,
        distance=result.distance
    )

@app.delete("/delete_face/{nombre}")
async def delete_face(nombre: str):
    """
    Elimina un rostro registrado del sistema.
    
    - **nombre**: Nombre de la persona a eliminar
    """
    global known_face_encodings, known_face_names
    
    if nombre not in known_face_names:
        raise HTTPException(status_code=404, detail=f"No se encontró a '{nombre}' en los registros.")
    
    # Eliminar de la base de datos
    if not delete_face_from_db(nombre):
        raise HTTPException(status_code=404, detail=f"No se pudo eliminar a '{nombre}' de la base de datos.")
    
    # Eliminar de la memoria
    index = known_face_names.index(nombre)
    known_face_names.pop(index)
    known_face_encodings.pop(index)
    
    return {"message": f"'{nombre}' ha sido eliminado del sistema."}

@app.get("/list_known_faces/")
async def list_known_faces():
    """Lista todos los rostros conocidos en el sistema."""
    return {"known_faces": known_face_names}

@app.get("/status/")
async def get_status():
    """Obtiene el estado actual de la API."""
    return {
        "status": "running",
        "loaded_faces_count": len(known_face_names),
        "database_connection": get_db_status(),
        "message": "API de reconocimiento facial lista para recibir solicitudes."
    }

@app.post("/refresh_faces/")
async def refresh_faces():
    """Recarga los rostros conocidos desde la base de datos."""
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = load_known_faces_from_db()
    return {"message": f"Rostros recargados. Total: {len(known_face_names)}"}

# Para ejecutar la aplicación directamente con Python
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
