from fastapi import FastAPI, HTTPException, Form, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
import os
import uvicorn
import numpy as np
from typing import List, Optional
from pathlib import Path
import asyncio
import logging

# Import our modules
from .database import (
    get_db_connection, 
    load_known_faces_from_db, 
    delete_face_from_db,
    save_face_to_db,
    get_db_status
)
from .photo_cleaner import run_photo_cleanup
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    print("\n" + "="*50)
    print("Iniciando API de Reconocimiento Facial...")
    print("="*50)
    
    # Verificar y mostrar la ruta de las fotos de referencia
    print(f"\nBuscando fotos de referencia en: {IMAGE_FOLDER.absolute()}")
    if not IMAGE_FOLDER.exists():
        print(f"¡Error! No se encontró el directorio: {IMAGE_FOLDER}")
    else:
        image_files = [f for f in IMAGE_FOLDER.glob('*') 
                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"Se encontraron {len(image_files)} imágenes de referencia.")
    
    # Cargar los rostros conocidos desde la base de datos
    try:
        known_face_encodings, known_face_names = load_known_faces_from_db()
        logger.info(f"[{len(known_face_names)}] rostros cargados desde la base de datos.")
        if known_face_names:
            logger.info(f"Personas reconocidas: {', '.join(known_face_names)}")
        
        # Start the background cleanup task
        asyncio.create_task(run_photo_cleanup())
        
    except Exception as e:
        logger.error(f"Error al cargar los rostros desde la base de datos: {e}")
        raise
    
    # Registrar automáticamente las imágenes en la carpeta reference_photos
    print("\nProcesando imágenes de referencia...")
    registered_count = 0
    error_count = 0
    
    for img_path in IMAGE_FOLDER.glob('*'):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        nombre = img_path.stem
        print(f"\nProcesando: {img_path.name}")
        
        try:
            # Leer la imagen
            with open(img_path, 'rb') as f:
                image_data = f.read()
            
            # Verificar si la imagen ya está registrada
            if nombre in known_face_names:
                print(f"  - {nombre} ya está registrado. Actualizando...")
            
            # Procesar y registrar el rostro
            print("  - Extrayendo características faciales...")
            result = process_and_register_face(
                image_data=image_data,
                nombre=nombre,
                known_face_encodings=known_face_encodings,
                known_face_names=known_face_names
            )
            
            # Guardar el embedding en la base de datos
            print("  - Guardando en la base de datos...")
            face_encoding = np.array(result['face_encoding'])
            save_face_to_db(nombre, face_encoding)
            
            print(f"  - ✓ {nombre} registrado/actualizado exitosamente.")
            registered_count += 1
            
        except Exception as e:
            error_count += 1
            print(f"  - ✗ Error al procesar {img_path.name}: {str(e)}")
    
    # Mostrar resumen
    print("\n" + "="*50)
    print("Resumen de inicio:")
    print(f"- Imágenes de referencia encontradas: {len(list(IMAGE_FOLDER.glob('*.jp*g'))) + len(list(IMAGE_FOLDER.glob('*.png')))}")
    print(f"- Rostros registrados exitosamente: {registered_count}")
    print(f"- Errores durante el registro: {error_count}")
    print(f"- Total de rostros cargados: {len(known_face_names)}")
    print("="*50 + "\n")

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
    
    # Guardar la imagen en el sistema de archivos
    try:
        # Normalizar el nombre para usarlo como nombre de archivo
        safe_name = "".join(c if c.isalnum() or c in ' .-_' else '_' for c in nombre)
        file_extension = os.path.splitext(file.filename)[1] or '.jpg'
        file_path = os.path.join(IMAGE_FOLDER, f"{safe_name}{file_extension}")
        
        # Guardar el archivo
        with open(file_path, "wb") as f:
            f.write(contents)
        logger.info(f"Imagen guardada en: {file_path}")
    except Exception as e:
        logger.error(f"Error al guardar la imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar la imagen: {e}")
    
    # Procesar y registrar el rostro
    try:
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
    except Exception as e:
        # Si hay un error, intentar eliminar la imagen guardada
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Imagen eliminada debido a un error: {file_path}")
        except Exception as cleanup_error:
            logger.error(f"Error al limpiar la imagen después de un error: {cleanup_error}")
        raise

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
    
    - **nombre**: Nombre de la persona a eliminar (no sensible a mayúsculas/minúsculas)
    """
    global known_face_encodings, known_face_names
    
    # Normalizar el nombre (quitar espacios extra)
    nombre = nombre.strip()
    
    # Primero intentar eliminar directamente de la base de datos
    try:
        # Obtener conexión a la base de datos
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Buscar el nombre exacto en la base de datos (insensible a mayúsculas/minúsculas)
        cur.execute("SELECT nombre FROM mis_personas WHERE LOWER(nombre) = LOWER(%s)", (nombre,))
        result = cur.fetchone()
        
        if not result:
            # Si no se encuentra, obtener la lista de nombres para el mensaje de error
            cur.execute("SELECT nombre FROM mis_personas")
            all_names = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró a '{nombre}' en la base de datos. Nombres registrados: {', '.join(all_names) if all_names else 'Ninguno'}"
            )
        
        # Obtener el nombre exacto como está en la base de datos
        actual_name = result[0]
        
        # Eliminar de la base de datos
        cur.execute("DELETE FROM mis_personas WHERE nombre = %s RETURNING nombre", (actual_name,))
        deleted = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()
        
        if not deleted:
            raise HTTPException(status_code=500, detail=f"Error al eliminar a '{actual_name}' de la base de datos.")
        
        # Actualizar la lista en memoria
        if actual_name in known_face_names:
            index = known_face_names.index(actual_name)
            known_face_names.pop(index)
            known_face_encodings.pop(index)
        
        return {"message": f"'{actual_name}' eliminado exitosamente del sistema."}
        
    except psycopg2.Error as e:
        if 'conn' in locals() and conn:
            conn.rollback()
        logger.error(f"Error en la base de datos al eliminar: {e}")
        raise HTTPException(status_code=500, detail=f"Error en la base de datos: {e}")
    except Exception as e:
        logger.error(f"Error inesperado al eliminar: {e}")
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")

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
