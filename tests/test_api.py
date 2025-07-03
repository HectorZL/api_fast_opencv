import os
import pytest
from fastapi.testclient import TestClient
import numpy as np
from app.main import app
from app.core.face_recognition_service import face_service

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Bienvenido a la API de Reconocimiento Facial" in response.json()["message"]

def test_register_face():
    # Necesitarías una imagen de prueba para esto
    test_image_path = "tests/test_face.jpg"
    if not os.path.exists(test_image_path):
        pytest.skip("No se encontró la imagen de prueba")
    
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/face_auth/register_face/",
            files={"file": ("test_face.jpg", f, "image/jpeg")},
            data={"nombre": "test_user"}
        )
    
    assert response.status_code == 200
    assert "registrado" in response.json()["message"].lower()

def test_recognize_face():
    test_image_path = "tests/test_face.jpg"
    if not os.path.exists(test_image_path):
        pytest.skip("No se encontró la imagen de prueba")
    
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/api/v1/face_auth/recognize_face/",
            files={"file": ("test_face.jpg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    assert "name" in response.json()

def test_list_known_faces():
    response = client.get("/api/v1/face_auth/list_known_faces/")
    assert response.status_code == 200
    assert isinstance(response.json()["known_faces"], list)

def test_get_status():
    response = client.get("/api/v1/face_auth/status/")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "loaded_faces_count" in response.json()

def test_delete_face():
    # Primero nos aseguramos de que exista un usuario de prueba
    test_username = "test_user"
    
    # Intentamos eliminar el usuario de prueba
    response = client.delete(f"/api/v1/face_auth/delete_face/{test_username}")
    
    # La respuesta puede ser 200 (si existía) o 404 (si no existía)
    assert response.status_code in [200, 404]
