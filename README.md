# API de Reconocimiento Facial

API desarrollada con FastAPI para el reconocimiento facial, que permite registrar, reconocer y eliminar rostros utilizando PostgreSQL para el almacenamiento.

## Estructura del Proyecto

```
API_FAST/
├── app/
│   ├── __init__.py
│   ├── main.py               # Punto de entrada de la aplicación
│   ├── config.py             # Configuraciones de la aplicación
│   │
│   ├── api/                  # API Routes
│   │   ├── __init__.py
│   │   └── v1/               # Versión 1 de la API
│   │       ├── __init__.py
│   │       ├── api.py        # Router principal de la API v1
│   │       └── endpoints/    # Endpoints agrupados por dominio
│   │           ├── __init__.py
│   │           └── face_auth.py  # Endpoints de autenticación facial
│   │
│   ├── core/                 # Lógica principal
│   │   ├── __init__.py
│   │   └── face_recognition_service.py  # Servicio de reconocimiento facial
│   │
│   ├── db/                   # Base de datos
│   │   ├── __init__.py
│   │   └── database.py       # Conexión a la base de datos
│   │
│   └── schemas/              # Esquemas Pydantic
│       ├── __init__.py
│       └── face.py           # Esquemas para reconocimiento facial
│
├── tests/                    # Pruebas
├── .env.example              # Ejemplo de variables de entorno
├── requirements.txt          # Dependencias
└── README.md                 # Este archivo
```

## Requisitos

- Python 3.8+
- PostgreSQL
- OpenCV
- face-recognition

## Instalación

1. Clona el repositorio:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd API_FAST
   ```

2. Crea un entorno virtual y actívalo:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Crea un archivo `.env` basado en `.env.example` y configura las variables de entorno.

## Configuración

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```
DB_HOST=localhost
DB_NAME=mi_base_de_datos
DB_USER=mi_usuario
DB_PASSWORD=mi_contraseña_segura
SECRET_KEY=tu_clave_secreta_muy_larga
```

## Base de Datos

Asegúrate de tener PostgreSQL instalado y crea una base de datos. Luego, ejecuta el siguiente script SQL para crear la tabla necesaria:

```sql
CREATE TABLE mis_personas (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    embedding REAL[]
);
```

## Ejecución

Para iniciar el servidor de desarrollo:

```bash
uvicorn app.main:app --reload
```

La API estará disponible en `http://127.0.0.1:8000`

## Documentación de la API

- Documentación interactiva: `http://127.0.0.1:8000/docs`
- Documentación alternativa: `http://127.0.0.1:8000/redoc`

## Endpoints Principales

- `POST /api/v1/face_auth/recognize_face/` - Reconoce un rostro en una imagen
- `POST /api/v1/face_auth/register_face/` - Registra un nuevo rostro
- `DELETE /api/v1/face_auth/delete_face/{nombre}` - Elimina un rostro registrado
- `GET /api/v1/face_auth/list_known_faces/` - Lista todos los rostros registrados
- `GET /api/v1/face_auth/status/` - Obtiene el estado del servicio

## Pruebas

Para ejecutar las pruebas:

```bash
pytest tests/
```

## Contribución

1. Haz un fork del proyecto
2. Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`)
3. Haz commit de tus cambios (`git commit -am 'Añade nueva característica'`)
4. Haz push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
