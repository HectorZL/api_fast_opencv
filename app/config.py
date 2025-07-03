from pydantic import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Database configuration
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_NAME: str = os.getenv("DB_NAME", "mi_base_de_datos")
    DB_USER: str = os.getenv("DB_USER", "mi_usuario")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "mi_contraseña_segura")
    
    # Application settings
    FACE_RECOGNITION_TOLERANCE: float = 0.6
    IMAGE_FOLDER: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference_photos")
    
    class Config:
        case_sensitive = True

settings = Settings()
