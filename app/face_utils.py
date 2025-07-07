import os
import cv2
import numpy as np
import face_recognition
from typing import Tuple, List, Optional
from fastapi import HTTPException

# Tolerancia para la comparación de rostros (ajustar según sea necesario, 0.6 es común)
FACE_RECOGNITION_TOLERANCE = 0.6

class FaceRecognitionResult:
    def __init__(self, name: str, is_known: bool, distance: Optional[float] = None):
        self.name = name
        self.is_known = is_known
        self.distance = distance

def process_image(image_data: bytes) -> np.ndarray:
    """Procesa los datos de la imagen y la convierte a formato RGB."""
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def extract_face_embedding(image_rgb: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """Extrae el embedding facial de una imagen RGB."""
    face_locations = face_recognition.face_locations(image_rgb)
    if not face_locations:
        return False, None
    
    # Usamos solo el primer rostro detectado
    face_encoding = face_recognition.face_encodings(image_rgb, face_locations)[0]
    return True, face_encoding

def recognize_face(
    test_face_encoding: np.ndarray, 
    known_face_encodings: List[np.ndarray], 
    known_face_names: List[str]
) -> FaceRecognitionResult:
    """
    Compara un rostro con una lista de rostros conocidos.
    
    Args:
        test_face_encoding: Embedding del rostro a reconocer
        known_face_encodings: Lista de embeddings de rostros conocidos
        known_face_names: Nombres correspondientes a los rostros conocidos
        
    Returns:
        FaceRecognitionResult con el resultado del reconocimiento
    """
    if not known_face_encodings:
        return FaceRecognitionResult(
            name="No hay rostros de referencia cargados en memoria.",
            is_known=False
        )
    
    # Convertir las listas a arrays de NumPy si no lo están ya
    test_face_encoding = np.array(test_face_encoding)
    known_face_encodings = [np.array(enc) for enc in known_face_encodings]
    
    # Realizar la comparación
    face_distances = face_recognition.face_distance(
        known_face_encodings, 
        test_face_encoding
    )
    
    matches = face_distances <= FACE_RECOGNITION_TOLERANCE
    
    if any(matches):
        best_match_index = np.argmin(face_distances)
        return FaceRecognitionResult(
            name=known_face_names[best_match_index],
            is_known=True,
            distance=float(face_distances[best_match_index])
        )
    else:
        # Si no hay coincidencias, devolvemos el más cercano aunque supere la tolerancia
        best_match_index = np.argmin(face_distances)
        return FaceRecognitionResult(
            name="Desconocido",
            is_known=False,
            distance=float(face_distances[best_match_index])
        )

def process_and_register_face(
    image_data: bytes, 
    nombre: str,
    known_face_encodings: List[np.ndarray],
    known_face_names: List[str]
) -> dict:
    """
    Procesa una imagen, extrae el rostro y lo registra en el sistema.
    
    Returns:
        dict con el resultado de la operación
    """
    try:
        # Procesar la imagen
        rgb_image = process_image(image_data)
        
        # Extraer el rostro
        success, face_encoding = extract_face_embedding(rgb_image)
        if not success or face_encoding is None:
            raise HTTPException(status_code=400, detail="No se detectó ningún rostro en la imagen.")
        
        # Verificar si el nombre ya existe
        if nombre in known_face_names:
            # Actualizar el rostro existente
            index = known_face_names.index(nombre)
            known_face_encodings[index] = face_encoding
        else:
            # Agregar el nuevo rostro
            known_face_encodings.append(face_encoding)
            known_face_names.append(nombre)
        
        return {
            "message": f"'{nombre}' registrado/actualizado.",
            "name": nombre,
            "face_encoding": face_encoding.tolist()
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al procesar el rostro: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el rostro: {e}")
