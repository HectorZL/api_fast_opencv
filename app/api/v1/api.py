from fastapi import APIRouter
from .endpoints import face_auth

# Crear el router principal de la API v1
api_router = APIRouter()

# Incluir los routers de los endpoints
api_router.include_router(
    face_auth.router,
    prefix="/face_auth",
    tags=["Autenticación Facial"]
)
