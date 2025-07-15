import os
import cv2
import numpy as np
import face_recognition
from typing import Tuple, List, Optional, Dict
from fastapi import HTTPException

# Tolerancia para la comparación de rostros (menor = más estricto, 0.5 es más preciso que el valor por defecto de 0.6)
FACE_RECOGNITION_TOLERANCE = 0.25

# Dimensiones objetivo para el rostro
TARGET_FACE_SIZE = (100, 100)

# Umbral de nitidez (varianza de Laplaciano)
SHARPNESS_THRESHOLD = 60.0

def crop_and_resize_face(image: np.ndarray, face_location: Tuple[int, int, int, int]) -> np.ndarray:
    """Recorta y redimensiona el rostro a tamaño estándar."""
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    
    # Redimensionar a tamaño estándar
    return cv2.resize(face_image, TARGET_FACE_SIZE, interpolation=cv2.INTER_AREA)

def calculate_sharpness(image: np.ndarray) -> float:
    """Calcula la nitidez de la imagen usando varianza de Laplaciano."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def apply_morphological_operations(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """Aplica operaciones morfológicas y devuelve el valor de apertura."""
    try:
        # Convertir a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplicar apertura morfológica
        kernel = np.ones((3,3), np.uint8)
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Calcular el valor de apertura
        aperture_value = np.mean(opened - gray)
        
        return opened, aperture_value
    except Exception as e:
        print(f"Error en operaciones morfológicas: {e}")
        return image, 0.0
    
class FaceRecognitionResult:
    def __init__(self, name: str, is_known: bool, distance: Optional[float] = None):
        self.name = name
        self.is_known = is_known
        self.distance = distance

def process_image(image_data: bytes) -> Tuple[np.ndarray, dict]:
    """
    Procesa los datos de la imagen, detecta y optimiza el rostro.
    
    Args:
        image_data: Datos binarios de la imagen
        
    Returns:
        tuple: (imagen procesada en escala de grises, metadatos de procesamiento)
    """
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")
    
    # Inicializar metadatos
    metadata = {
        'original_size': image.shape[:2],
        'face_detected': False,
        'face_location': None,
        'face_size': None,
        'sharpness': 0.0,
        'applied_operations': [],
        'aperture_value': 0.0
    }
    
    try:
        # Detectar rostro
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            raise HTTPException(status_code=400, detail="No se detectó ningún rostro en la imagen.")
            
        # Usar el primer rostro detectado
        face_location = face_locations[0]
        metadata['face_detected'] = True
        metadata['face_location'] = face_location
        
        # Recortar y redimensionar el rostro
        face_image = crop_and_resize_face(image, face_location)
        metadata['face_size'] = TARGET_FACE_SIZE
        
        # Convertir a escala de grises
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Calcular nitidez
        sharpness = calculate_sharpness(gray_face)
        metadata['sharpness'] = float(sharpness)
        
        # Si la nitidez es baja, aplicar operaciones morfológicas
        if sharpness < SHARPNESS_THRESHOLD:
            metadata['applied_operations'].append('morphological_operations')
            gray_face, aperture_value = apply_morphological_operations(face_image)
            metadata['aperture_value'] = float(aperture_value)
            
        return gray_face, metadata
        
    except Exception as e:
        print(f"Error en el procesamiento de imagen: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

def extract_face_embedding(image_gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray], dict]:
    """
    Extrae el embedding facial de una imagen en escala de grises con múltiples intentos de detección.
    
    Args:
        image_gray: Imagen en escala de grises
        
    Returns:
        tuple: (éxito, embedding, metadatos)
    """
    if image_gray is None or image_gray.size == 0:
        return False, None, {'face_detected': False, 'error': 'Imagen vacía o inválida'}
    
    # Convertir a RGB para face_recognition
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    
    # Mejorar el contraste para una mejor detección
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    # Métodos de detección a probar (en orden de precisión)
    detection_methods = [
        {'model': 'hog', 'number_of_times_to_upsample': 2},
        {'model': 'hog', 'number_of_times_to_upsample': 1},
        {'model': 'cnn', 'number_of_times_to_upsample': 1}
    ]
    
    best_face = None
    best_quality = -1
    
    for method in detection_methods:
        try:
            # Intentar detectar rostros en la imagen mejorada
            face_locations = face_recognition.face_locations(
                enhanced_image,
                model=method['model'],
                number_of_times_to_upsample=method['number_of_times_to_upsample']
            )
            
            if not face_locations:
                continue
                
            # Tomar solo el rostro con mejor calidad
            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = enhanced_image[top:bottom, left:right]
                
                # Calcular calidad del rostro (tamaño y nitidez)
                face_size = (right - left) * (bottom - top)
                face_sharpness = cv2.Laplacian(cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
                quality = face_size * (1 + face_sharpness/100)  # Ponderar tamaño y nitidez
                
                if quality > best_quality:
                    best_quality = quality
                    best_face = face_location
        except Exception as e:
            print(f"Error en detección con método {method}: {e}")
            continue
    
    if best_face is not None:
        try:
            # Extraer el embedding del mejor rostro detectado
            face_encodings = face_recognition.face_encodings(
                enhanced_image,
                [best_face],
                num_jitters=3,  # Aumentar jittering para mayor precisión
                model='large'   # Usar el modelo large para mayor precisión
            )
            
            if face_encodings:
                return True, face_encodings[0], {
                    'face_detected': True,
                    'face_location': best_face,
                    'quality_score': best_quality,
                    'detection_method': 'best_of_all'
                }
        except Exception as e:
            print(f"Error al extraer embedding: {e}")
    
    # Si llegamos aquí, ningún método funcionó
    return False, None, {
        'face_detected': False,
        'error': 'No se pudo detectar ningún rostro en la imagen con ningún método'
    }

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

def is_face_duplicate(
    new_encoding: np.ndarray, 
    known_face_encodings: List[np.ndarray],
    known_face_names: List[str],
    tolerance: float = 0.5  # Más estricto que el valor por defecto
) -> Tuple[bool, Optional[str], float]:
    """
    Verifica si un rostro ya está registrado comparando con los existentes.
    
    Args:
        new_encoding: Embedding del nuevo rostro
        known_face_encodings: Lista de embeddings conocidos
        known_face_names: Nombres correspondientes a los embeddings
        tolerance: Tolerancia para la comparación (menor = más estricto)
        
    Returns:
        tuple: (es_duplicado, nombre_duplicado, distancia)
    """
    if not known_face_encodings:
        return False, None, float('inf')
    
    # Calcular distancias con todos los rostros conocidos
    distances = face_recognition.face_distance(known_face_encodings, new_encoding)
    min_distance = np.min(distances)
    min_index = np.argmin(distances)
    
    # Si la distancia es muy pequeña, es claramente un duplicado
    if min_distance < tolerance * 0.8:  # Más estricto para coincidencias claras
        return True, known_face_names[min_index], float(min_distance)
    
    # Si está en el rango dudoso, aplicar verificación adicional
    if min_distance < tolerance:
        # Verificar con otro método de comparación
        matches = face_recognition.compare_faces(
            known_face_encodings, 
            new_encoding, 
            tolerance=tolerance * 0.9  # Un poco más estricto para la verificación
        )
        if any(matches):
            return True, known_face_names[min_index], float(min_distance)
    
    return False, None, float(min_distance)

def process_and_register_face(
    image_data: bytes, 
    nombre: str,
    known_face_encodings: List[np.ndarray],
    known_face_names: List[str],
    update_existing: bool = False
) -> dict:
    """
    Procesa y registra un rostro, evitando duplicados.
    
    Args:
        image_data: Datos binarios de la imagen
        nombre: Nombre de la persona
        known_face_encodings: Lista de embeddings conocidos
        known_face_names: Nombres correspondientes a los embeddings
        update_existing: Si es True, actualiza el registro existente si el nombre ya existe
        
    Returns:
        dict: Resultado del registro
    """
    try:
        # Procesar la imagen
        rgb_image, metadata = process_image(image_data)
        
        # Extraer el rostro
        success, face_encoding, face_metadata = extract_face_embedding(rgb_image)
        if not success or face_encoding is None:
            error_msg = face_metadata.get('error', 'No se detectó ningún rostro en la imagen.')
            if metadata.get('is_blurry', False):
                error_msg += f" La imagen parece estar borrosa (varianza Laplaciano: {metadata.get('blur_variance', 0):.2f})."
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Verificar si el rostro ya está registrado
        is_duplicate, duplicate_name, distance = is_face_duplicate(
            face_encoding, 
            known_face_encodings,
            known_face_names
        )
        
        # Manejar duplicados
        if is_duplicate:
            if duplicate_name.lower() == nombre.lower() and not update_existing:
                return {
                    "message": f"El rostro de '{nombre}' ya está registrado.",
                    "name": nombre,
                    "is_duplicate": True,
                    "duplicate_name": duplicate_name,
                    "distance": distance,
                    "metadata": {**metadata, **face_metadata}
                }
            elif update_existing and duplicate_name.lower() == nombre.lower():
                # Actualizar el registro existente
                index = known_face_names.index(duplicate_name)
                known_face_encodings[index] = face_encoding
                action = "actualizado"
            else:
                # Rostro duplicado pero con nombre diferente
                return {
                    "message": f"Este rostro parece ser de '{duplicate_name}' (distancia: {distance:.4f}).",
                    "name": nombre,
                    "is_duplicate": True,
                    "duplicate_name": duplicate_name,
                    "distance": distance,
                    "metadata": {**metadata, **face_metadata}
                }
        else:
            # No es un duplicado, agregar nuevo registro
            known_face_encodings.append(face_encoding)
            known_face_names.append(nombre)
            action = "registrado"
        
        # Devolver resultado
        return {
            "message": f"'{nombre}' {action} exitosamente.",
            "name": nombre,
            "is_duplicate": False,
            "distance": distance if is_duplicate else float('inf'),
            "face_encoding": face_encoding.tolist(),
            "metadata": {**metadata, **face_metadata}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al procesar el rostro: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el rostro: {e}")