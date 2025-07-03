from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
import os
from ....core.face_recognition_service import face_service
from ....schemas.face import (
    RecognitionResult, 
    FaceRegisterRequest, 
    FaceRegisterResponse,
    FaceDeleteResponse,
    StatusResponse
)
from fastapi import Form

router = APIRouter()

@router.post("/recognize_face/", response_model=RecognitionResult)
async def recognize_face(file: UploadFile = File(...)):
    """
    Compara una nueva foto subida con los rostros cargados de la base de datos.
    Retorna el nombre del rostro reconocido o "Desconocido".
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Formato de archivo no válido. Se esperaba una imagen.")
    
    # Leer la imagen
    contents = await file.read()
    
    # Usar el servicio para reconocer el rostro
    result = face_service.recognize_face(contents)
    return result

@router.post("/register_face/", response_model=FaceRegisterResponse)
async def register_face(
    nombre: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Registra un nuevo rostro en el sistema.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Formato de archivo no válido. Se esperaba una imagen.")
    
    # Leer la imagen
    contents = await file.read()
    
    # Usar el servicio para registrar el rostro
    result = await face_service.register_face(nombre, contents)
    return result

@router.post("/register_face_from_local_path/", response_model=FaceRegisterResponse)
async def register_face_from_local_path(
    nombre: str = Form(...),
    local_image_path: str = Form(...)
):
    """
    Registra un rostro desde una ruta de archivo local.
    Principalmente para uso en desarrollo/pruebas.
    """
    if not os.path.exists(local_image_path) or not os.path.isfile(local_image_path):
        raise HTTPException(status_code=400, detail=f"La ruta de imagen '{local_image_path}' no es válida.")
    
    # Leer la imagen
    with open(local_image_path, 'rb') as f:
        contents = f.read()
    
    # Usar el servicio para registrar el rostro
    result = await face_service.register_face(nombre, contents)
    return result

@router.delete("/delete_face/{nombre}", response_model=FaceDeleteResponse)
async def delete_face(nombre: str):
    """
    Elimina un rostro registrado por su nombre.
    """
    result = face_service.delete_face(nombre)
    return result

@router.get("/list_known_faces/")
async def list_known_faces():
    """
    Lista todos los nombres de las personas con rostros registrados.
    """
    return {"known_faces": face_service.known_face_names}

@router.get("/status/", response_model=StatusResponse)
async def get_status():
    """
    Obtiene el estado actual del servicio.
    """
    return face_service.get_status()

@router.post("/refresh_db_faces/")
async def refresh_db_faces():
    """
    Recarga los rostros de referencia desde la base de datos.
    """
    face_service.load_known_faces_from_db()
    return {"message": f"Rostros recargados desde la DB. Total: {len(face_service.known_face_names)}"}
