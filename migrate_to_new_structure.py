"""
Script para migrar desde la estructura antigua a la nueva.
Este script debe ejecutarse una sola vez.
"""
import os
import shutil
import sys
from pathlib import Path

# Asegurarse de que el directorio de trabajo sea el correcto
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Crear la carpeta reference_photos si no existe
reference_photos_dir = Path("reference_photos")
reference_photos_dir.mkdir(exist_ok=True)

# Copiar el archivo .env si no existe
if not os.path.exists(".env") and os.path.exists(".env.example"):
    shutil.copy2(".env.example", ".env")
    print("Archivo .env creado a partir de .env.example. Por favor, configura las variables de entorno.")

print("Migración completada. Por favor, sigue estos pasos:")
print("1. Configura el archivo .env con tus credenciales")
print("2. Instala las dependencias: pip install -r requirements.txt")
print("3. Ejecuta la aplicación: uvicorn app.main:app --reload")
print("4. Accede a la documentación en: http://127.0.0.1:8000/docs")
