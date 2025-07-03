import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.v1.api import api_router
from .core.face_recognition_service import face_service
from .config import settings

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Reconocimiento Facial",
    description="API para autenticación mediante reconocimiento facial",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir los routers de la API
app.include_router(api_router, prefix="/api/v1")

# Evento de inicio: cargar rostros conocidos
@app.on_event("startup")
async def startup_event():
    """Se ejecuta cuando la aplicación FastAPI arranca."""
    print("Iniciando API de Reconocimiento Facial...")
    
    # Cargar rostros conocidos desde la base de datos
    face_service.load_known_faces_from_db()
    
    # Cargar automáticamente imágenes de la carpeta de referencia (opcional)
    if os.path.exists(settings.IMAGE_FOLDER):
        print(f"Buscando imágenes en {settings.IMAGE_FOLDER}...")
        for filename in os.listdir(settings.IMAGE_FOLDER):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                nombre = os.path.splitext(filename)[0]
                local_image_path = os.path.join(settings.IMAGE_FOLDER, filename)
                try:
                    # Registrar automáticamente la imagen
                    with open(local_image_path, 'rb') as f:
                        contents = f.read()
                    await face_service.register_face(nombre, contents)
                    print(f"Registrado automáticamente: {nombre}")
                except Exception as e:
                    print(f"Error al registrar {nombre}: {e}")
    else:
        print(f"Advertencia: No se encontró la carpeta de imágenes en {settings.IMAGE_FOLDER}")

# Ruta raíz
@app.get("/")
async def root():
    """Ruta raíz que devuelve un mensaje de bienvenida."""
    return {
        "message": "Bienvenido a la API de Reconocimiento Facial",
        "version": "1.0.0",
        "docs": "/docs"
    }
