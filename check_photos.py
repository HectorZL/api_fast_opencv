import os
import cv2
import numpy as np
import face_recognition
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de rutas
BASE_DIR = Path(__file__).resolve().parent
IMAGE_FOLDER = BASE_DIR / "reference_photos"

print(f"Verificando imágenes en: {IMAGE_FOLDER}")

# Verificar si el directorio existe
if not IMAGE_FOLDER.exists():
    print(f"Error: No se encontró el directorio {IMAGE_FOLDER}")
    exit(1)

# Obtener lista de archivos de imagen
image_files = [f for f in IMAGE_FOLDER.glob('*') 
              if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

if not image_files:
    print("No se encontraron archivos de imagen (.jpg, .jpeg, .png) en el directorio.")
    exit(1)

print(f"\nSe encontraron {len(image_files)} imágenes:")
for i, img_path in enumerate(image_files, 1):
    print(f"{i}. {img_path.name}")

# Procesar cada imagen
print("\nProcesando imágenes...")
for img_path in image_files:
    try:
        print(f"\nProcesando: {img_path.name}")
        
        # Leer la imagen
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Error: No se pudo leer la imagen {img_path.name}")
            continue
            
        print(f"  Tamaño: {image.shape[1]}x{image.shape[0]} píxeles")
        
        # Convertir a RGB (face_recognition usa RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar caras
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            print("  No se detectaron caras en la imagen")
            continue
            
        print(f"  Se detectaron {len(face_locations)} cara(s)")
        
        # Obtener el primer rostro
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        print(f"  Tamaño del embedding: {len(face_encoding)} dimensiones")
        
    except Exception as e:
        print(f"  Error al procesar {img_path.name}: {str(e)}")

print("\nAnálisis completado.")
