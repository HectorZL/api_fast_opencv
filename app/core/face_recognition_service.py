import os
import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Optional, Dict, Any
from fastapi import HTTPException

class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.tolerance: float = 0.6
        
    def load_known_faces_from_db(self):
        """Carga nombres y embeddings desde la base de datos."""
        self.known_face_encodings = []
        self.known_face_names = []
        
        from ..db.database import get_db_connection
        import psycopg2
        
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT nombre, embedding FROM mis_personas WHERE embedding IS NOT NULL")
            rows = cur.fetchall()
            
            for nombre, embedding_data in rows:
                self.known_face_names.append(nombre)
                self.known_face_encodings.append(np.array(embedding_data))
            
            print(f"[{len(self.known_face_names)}] rostros (embeddings) cargados de la base de datos.")
            if self.known_face_names:
                print(f"Personas cargadas: {', '.join(self.known_face_names)}")
                
        except Exception as e:
            print(f"Error al cargar rostros de la DB: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error en la base de datos al cargar rostros: {e}"
            )
        finally:
            if conn:
                cur.close()
                conn.close()
    
    async def register_face(self, nombre: str, image_data: bytes) -> Dict[str, Any]:
        """Registra un nuevo rostro en la base de datos."""
        # Procesar la imagen
        success, embedding = self._process_image(image_data)
        if not success or embedding is None:
            raise HTTPException(status_code=400, detail="No se pudo procesar la imagen o no se detectó un rostro.")
        
        # Guardar en la base de datos
        from ..db.database import get_db_connection
        import psycopg2
        
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Verificar si ya existe
            cur.execute("SELECT 1 FROM mis_personas WHERE nombre = %s", (nombre,))
            if cur.fetchone():
                # Actualizar existente
                cur.execute(
                    "UPDATE mis_personas SET embedding = %s WHERE nombre = %s",
                    (embedding.tolist(), nombre)
                )
            else:
                # Insertar nuevo
                cur.execute(
                    "INSERT INTO mis_personas (nombre, embedding) VALUES (%s, %s)",
                    (nombre, embedding.tolist())
                )
            
            conn.commit()
            self.load_known_faces_from_db()  # Recargar desde DB
            
            return {
                "message": f"'{nombre}' registrado/actualizado correctamente.",
                "name": nombre
            }
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            print(f"Error en la base de datos: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error en la base de datos: {e}"
            )
        finally:
            if conn:
                cur.close()
                conn.close()
    
    def recognize_face(self, image_data: bytes) -> Dict[str, Any]:
        """Reconoce un rostro en la imagen proporcionada."""
        # Procesar la imagen
        success, test_face_encoding = self._process_image(image_data)
        if not success or test_face_encoding is None:
            return {"name": "No Rostro Detectado", "is_known": False, "distance": None}
        
        if not self.known_face_encodings:
            return {"name": "No hay rostros de referencia cargados en memoria.", "is_known": False, "distance": None}
        
        # Realizar la comparación
        matches = face_recognition.compare_faces(
            self.known_face_encodings, 
            test_face_encoding, 
            tolerance=self.tolerance
        )
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, 
            test_face_encoding
        )
        
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            return {
                "name": self.known_face_names[best_match_index],
                "is_known": True,
                "distance": float(face_distances[best_match_index])
            }
        else:
            return {
                "name": "Desconocido",
                "is_known": False,
                "distance": float(face_distances[best_match_index])
            }
    
    def delete_face(self, nombre: str) -> Dict[str, str]:
        """Elimina un rostro de la base de datos."""
        from ..db.database import get_db_connection
        import psycopg2
        
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            
            # Verificar si existe
            cur.execute("SELECT 1 FROM mis_personas WHERE nombre = %s", (nombre,))
            if not cur.fetchone():
                raise HTTPException(
                    status_code=404, 
                    detail=f"Persona con nombre '{nombre}' no encontrada."
                )
            
            # Eliminar
            cur.execute("DELETE FROM mis_personas WHERE nombre = %s", (nombre,))
            conn.commit()
            
            # Recargar desde DB
            self.load_known_faces_from_db()
            
            return {"message": f"Persona '{nombre}' eliminada exitosamente de la DB."}
            
        except HTTPException:
            raise
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            print(f"Error en la base de datos al eliminar: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error en la base de datos al eliminar: {e}"
            )
        finally:
            if conn:
                cur.close()
                conn.close()
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado del servicio."""
        db_status = "Desconectado"
        from ..db.database import get_db_connection
        import psycopg2
        
        conn = None
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            db_status = "Conectado"
        except Exception as e:
            db_status = f"Error de conexión: {str(e)}"
        finally:
            if conn:
                conn.close()
        
        return {
            "status": "running",
            "loaded_faces_count": len(self.known_face_names),
            "source": "Cargado desde la tabla 'mis_personas' de PostgreSQL",
            "database_connection": db_status,
            "message": "Listo para reconocer rostros."
        }
    
    def _process_image(self, image_data: bytes) -> Tuple[bool, Optional[np.ndarray]]:
        """Procesa una imagen y extrae el embedding facial."""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return False, None
                
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb)
            
            if not face_locations:
                return False, None
                
            # Devolver solo la primera cara detectada
            face_encoding = face_recognition.face_encodings(rgb, face_locations)[0]
            return True, face_encoding
            
        except Exception as e:
            print(f"Error al procesar la imagen: {e}")
            return False, None

# Instancia global del servicio
face_service = FaceRecognitionService()
