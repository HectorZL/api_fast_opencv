import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.face_recognition_service import face_service

@pytest.fixture(scope="module")
def test_client():
    """Fixture para el cliente de prueba."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(autouse=True)
def run_around_tests():
    """Fixture que se ejecuta antes y después de cada prueba."""
    # Código que se ejecuta antes de cada prueba
    print("\n--- Iniciando prueba ---")
    
    # Esto permite que la prueba se ejecute
    yield
    
    # Código que se ejecuta después de cada prueba
    print("--- Prueba finalizada ---\n")

@pytest.fixture
def test_image():
    """Fixture que proporciona una imagen de prueba."""
    test_image_path = "tests/test_face.jpg"
    with open(test_image_path, "rb") as f:
        return f.read()
