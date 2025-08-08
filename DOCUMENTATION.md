# Documentación Técnica: API de Reconocimiento Facial con FastAPI

## 📚 Tabla de Contenidos
1. [Introducción](#-introducción)
2. [Objetivos del Proyecto](#-objetivos-del-proyecto)
3. [Arquitectura del Sistema](#-arquitectura-del-sistema)
4. [Requisitos del Sistema](#-requisitos-del-sistema)
5. [Instalación y Configuración](#-instalación-y-configuración)
6. [Estructura del Proyecto](#-estructura-del-proyecto)
7. [Módulos y Componentes](#-módulos-y-componentes)
8. [Endpoints de la API](#-endpoints-de-la-api)
9. [Flujo de Trabajo](#-flujo-de-trabajo)
10. [Ejemplos de Uso](#-ejemplos-de-uso)
11. [Optimización de Rendimiento](#-optimización-de-rendimiento)
12. [Seguridad](#-seguridad)
13. [Manejo de Errores](#-manejo-de-errores)
14. [Pruebas](#-pruebas)
15. [Despliegue](#-despliegue)
16. [Mantenimiento](#-mantenimiento)
17. [Solución de Problemas](#-solución-de-problemas)
18. [Preguntas Frecuentes](#-preguntas-frecuentes)
19. [Contribución](#-contribución)
20. [Licencia](#-licencia)
21. [Contacto](#-contacto)

## 🌟 Introducción

La API de Reconocimiento Facial es una solución robusta y escalable desarrollada con FastAPI que permite el registro, identificación y gestión de rostros humanos en tiempo real. Este sistema está diseñado para ser utilizado en diversos escenarios que requieran autenticación biométrica, control de acceso o identificación de personas.

La implementación actual utiliza técnicas avanzadas de visión por computadora y aprendizaje automático para extraer características faciales únicas (embeddings) y compararlas con una base de datos de rostros previamente registrados. La arquitectura modular y el uso de estándares de la industria garantizan un alto rendimiento y facilidad de integración con otros sistemas.

## 🎯 Objetivos del Proyecto

### Objetivo Principal
Desarrollar un sistema de reconocimiento facial preciso, eficiente y fácil de integrar que permita la autenticación segura de usuarios mediante el análisis biométrico facial.

### Objetivos Específicos
1. **Precisión**: Lograr una alta tasa de reconocimiento con bajos índices de falsos positivos/negativos.
2. **Rendimiento**: Procesar las solicitudes en tiempo real con un uso eficiente de recursos.
3. **Escalabilidad**: Diseñar una arquitectura que pueda manejar desde decenas hasta miles de usuarios.
4. **Seguridad**: Implementar medidas robustas para proteger los datos biométricos.
5. **Facilidad de Uso**: Proporcionar una API intuitiva y documentación clara.
6. **Extensibilidad**: Permitir la adición de nuevas funcionalidades sin modificar el código existente.

## 🏗️ Arquitectura del Sistema

### Diagrama de Arquitectura
```
+----------------+     +----------------+     +-----------------+
|                |     |                |     |                 |
|  Cliente HTTP  |<--->|  API FastAPI   |<--->|  Base de Datos  |
|  (Frontend/App)|     |  (Backend)     |     |  PostgreSQL     |
|                |     |                |     |                 |
+----------------+     +----------------+     +-----------------+
                           |         ^
                           |         |
                           v         |
                     +------------------+
                     |  Procesamiento   |
                     |  de Imágenes     |
                     |  (OpenCV, dlib)  |
                     +------------------+
```

### Componentes Principales

1. **Capa de API (FastAPI)**
   - Manejo de solicitudes HTTP/HTTPS
   - Validación de datos de entrada
   - Autenticación y autorización
   - Enrutamiento de endpoints

2. **Capa de Procesamiento de Imágenes**
   - Detección de rostros
   - Extracción de características faciales
   - Comparación de rostros
   - Optimización de imágenes

3. **Capa de Almacenamiento**
   - Almacenamiento seguro de embeddings
   - Gestión de metadatos de usuarios
   - Consultas eficientes

4. **Capa de Seguridad**
   - Encriptación de datos
   - Gestión de tokens JWT
   - Protección contra ataques comunes

## 💻 Requisitos del Sistema

### Requisitos Mínimos
- **Sistema Operativo**: Windows 10/11, Linux (Ubuntu 20.04+), o macOS 10.15+
- **Procesador**: CPU de 64 bits con soporte AVX (para dlib)
- **Memoria RAM**: 4 GB (8 GB recomendado)
- **Almacenamiento**: 2 GB de espacio libre
- **Python**: 3.12 o superior
- **PostgreSQL**: 14 o superior

### Dependencias Principales
- **FastAPI**: Framework web moderno y rápido
- **face-recognition**: Biblioteca para reconocimiento facial
- **OpenCV**: Procesamiento de imágenes
- **NumPy**: Operaciones numéricas eficientes
- **PostgreSQL**: Base de datos relacional
- **SQLAlchemy**: ORM para Python
- **Python-multipart**: Manejo de carga de archivos
- **python-jose**: Implementación de JWT
- **python-dotenv**: Manejo de variables de entorno

## 🛠️ Instalación y Configuración

### 1. Clonar el Repositorio
```bash
git clone [URL_DEL_REPOSITORIO]
cd API_FAST
```

### 2. Configuración del Entorno Virtual
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalación de Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configuración de la Base de Datos
1. Instalar PostgreSQL si no está instalado
2. Crear una base de datos para el proyecto
3. Ejecutar el script de inicialización:
```sql
-- Crear la tabla para almacenar los rostros
CREATE TABLE mis_personas (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    embedding REAL[],
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Crear índice para búsquedas por nombre
CREATE INDEX idx_mis_personas_nombre ON mis_personas(nombre);

-- Crear extensión para operaciones con vectores (opcional, para PostgreSQL con extensión pgvector)
-- CREATE EXTENSION IF NOT EXISTS vector;
```

### 5. Configuración de Variables de Entorno
Crear un archivo `.env` en la raíz del proyecto con las siguientes variables:
```ini
# Configuración de la base de datos
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nombre_base_datos
DB_USER=usuario_postgres
DB_PASSWORD=tu_contraseña_segura

# Configuración de la aplicación
SECRET_KEY=tu_clave_secreta_muy_larga_y_segura
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Configuración de reconocimiento facial
FACE_RECOGNITION_TOLERANCE=0.5
FACE_DETECTION_CONFIDENCE=0.9
MAX_IMAGE_SIZE_MB=5
```

## 📂 Estructura del Proyecto

```
API_FAST/
├── app/                    # Código fuente principal
│   ├── __init__.py        # Inicialización de la aplicación
│   ├── main.py            # Punto de entrada de la aplicación
│   ├── config.py          # Configuración de la aplicación
│   │
│   ├── api/               # Definición de rutas de la API
│   │   ├── __init__.py
│   │   └── v1/            # Versión 1 de la API
│   │       ├── __init__.py
│   │       ├── api.py     # Router principal de la API v1
│   │       └── endpoints/ # Endpoints agrupados por dominio
│   │           ├── __init__.py
│   │           └── face_auth.py  # Endpoints de autenticación facial
│   │
│   ├── core/              # Lógica de negocio
│   │   ├── __init__.py
│   │   └── face_recognition_service.py  # Servicio de reconocimiento facial
│   │
│   ├── db/                # Configuración de base de datos
│   │   ├── __init__.py
│   │   ├── database.py    # Conexión a la base de datos
│   │   └── models.py      # Modelos de base de datos
│   │
│   ├── schemas/           # Esquemas Pydantic
│   │   ├── __init__.py
│   │   ├── face.py        # Esquemas para reconocimiento facial
│   │   └── user.py        # Esquemas para gestión de usuarios
│   │
│   └── utils/             # Utilidades
│       ├── __init__.py
│       ├── face_utils.py  # Funciones auxiliares para procesamiento facial
│       └── image_utils.py # Utilidades para procesamiento de imágenes
│
├── tests/                 # Pruebas unitarias y de integración
│   ├── __init__.py
│   ├── conftest.py        # Configuración de pytest
│   ├── test_api.py        # Pruebas de la API
│   └── test_models.py     # Pruebas de modelos
│
├── .env.example           # Ejemplo de archivo de variables de entorno
├── .gitignore             # Archivos ignorados por Git
├── requirements.txt       # Dependencias del proyecto
├── README.md              # Documentación básica
└── DOCUMENTATION.md       # Esta documentación detallada
```

## 🔧 Módulos y Componentes

### 1. Módulo de API (api/)

#### 1.1. Router Principal (api/v1/api.py)
- Configuración de rutas principales
- Inclusión de routers específicos
- Manejo de prefijos y etiquetas

#### 1.2. Endpoints de Autenticación Facial (api/v1/endpoints/face_auth.py)
- Registro de nuevos rostros
- Reconocimiento facial
- Eliminación de registros
- Listado de rostros registrados

### 2. Módulo de Procesamiento de Rostros (core/)

#### 2.1. Servicio de Reconocimiento Facial (core/face_recognition_service.py)
- Extracción de características faciales
- Comparación de rostros
- Gestión de la base de datos de rostros
- Validación de calidad de imágenes

### 3. Módulo de Base de Datos (db/)

#### 3.1. Conexión a la Base de Datos (db/database.py)
- Configuración de la conexión
- Pool de conexiones
- Manejo de sesiones

#### 3.2. Modelos de Datos (db/models.py)
- Definición de tablas
- Relaciones entre modelos
- Métodos de utilidad

### 4. Módulo de Utilidades (utils/)

#### 4.1. Utilidades de Procesamiento Facial (utils/face_utils.py)
- Detección de rostros
- Extracción de embeddings
- Comparación de rostros
- Validación de calidad

#### 4.2. Utilidades de Procesamiento de Imágenes (utils/image_utils.py)
- Redimensionamiento
- Normalización
- Mejora de contraste
- Conversión de formatos

## 🌐 Endpoints de la API

### 1. Registro de Rostro
```
POST /api/v1/face/register
Content-Type: multipart/form-data
```
**Parámetros:**
- `image`: Archivo de imagen (jpg, png)
- `name`: Nombre de la persona (opcional)
- `user_id`: ID de usuario (opcional)

**Respuesta Exitosa (200 OK):**
```json
{
    "status": "success",
    "data": {
        "id": 1,
        "name": "Juan Pérez",
        "face_id": "550e8400-e29b-41d4-a716-446655440000",
        "created_at": "2023-01-01T12:00:00Z"
    }
}
```

### 2. Reconocimiento de Rostro
```
POST /api/v1/face/recognize
Content-Type: multipart/form-data
```
**Parámetros:**
- `image`: Archivo de imagen (jpg, png)
- `threshold`: Umbral de confianza (opcional, por defecto 0.5)

**Respuesta Exitosa (200 OK):**
```json
{
    "status": "success",
    "data": {
        "is_recognized": true,
        "name": "Juan Pérez",
        "confidence": 0.92,
        "face_id": "550e8400-e29b-41d4-a716-446655440000",
        "processing_time_ms": 120
    }
}
```

### 3. Eliminación de Registro
```
DELETE /api/v1/face/{face_id}
```

**Respuesta Exitosa (200 OK):**
```json
{
    "status": "success",
    "message": "Rostro eliminado correctamente"
}
```

### 4. Listado de Rostros Registrados
```
GET /api/v1/face/list
```

**Respuesta Exitosa (200 OK):**
```json
{
    "status": "success",
    "data": [
        {
            "id": 1,
            "name": "Juan Pérez",
            "face_id": "550e8400-e29b-41d4-a716-446655440000",
            "created_at": "2023-01-01T12:00:00Z"
        },
        {
            "id": 2,
            "name": "María García",
            "face_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "created_at": "2023-01-02T10:30:00Z"
        }
    ]
}
```

## 🔄 Flujo de Trabajo

### 1. Registro de un Nuevo Rostro
1. El cliente envía una imagen al endpoint de registro
2. El servidor valida la imagen y detecta un rostro
3. Se extraen las características faciales (embedding)
4. Se verifica si el rostro ya está registrado
5. Si no es un duplicado, se guarda en la base de datos
6. Se devuelve un ID único para el rostro registrado

### 2. Reconocimiento de un Rostro
1. El cliente envía una imagen al endpoint de reconocimiento
2. El servidor detecta el rostro en la imagen
3. Se extraen las características faciales
4. Se comparan con los rostros registrados en la base de datos
5. Si se encuentra una coincidencia por encima del umbral, se devuelve la identificación
6. Si no se encuentra ninguna coincidencia, se devuelve "desconocido"

### 3. Eliminación de un Rostro
1. El cliente solicita la eliminación de un rostro por su ID
2. El servidor verifica los permisos
3. Se elimina el registro de la base de datos
4. Se confirma la eliminación

## 💡 Ejemplos de Uso

### 1. Registro de un Nuevo Usuario (Python)
```python
import requests

url = "http://localhost:8000/api/v1/face/register"
files = {'image': open('usuario.jpg', 'rb')}
data = {'name': 'Juan Pérez'}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### 2. Reconocimiento Facial (Python)
```python
import requests

url = "http://localhost:8000/api/v1/face/recognize"
files = {'image': open('foto_desconocido.jpg', 'rb')}
params = {'threshold': 0.5}

response = requests.post(url, files=files, params=params)
result = response.json()

if result['data']['is_recognized']:
    print(f"Persona identificada: {result['data']['name']} (Confianza: {result['data']['confidence']:.2f})")
else:
    print("No se pudo identificar a la persona")
```

### 3. Eliminación de un Registro (cURL)
```bash
curl -X DELETE "http://localhost:8000/api/v1/face/550e8400-e29b-41d4-a716-446655440000"
```

## ⚡ Optimización de Rendimiento

### 1. Caché de Embeddings
- Los embeddings de rostros conocidos se cargan en memoria al iniciar la aplicación
- Se actualizan periódicamente o cuando se realizan cambios

### 2. Procesamiento por Lotes
- Para múltiples reconocimientos, se pueden procesar en lotes
- Reducción del tiempo de procesamiento por imagen

### 3. Indexación en Base de Datos
- Uso de índices para búsquedas rápidas
- Particionamiento de tablas para grandes volúmenes de datos

### 4. Optimización de Imágenes
- Reducción de tamaño antes del procesamiento
- Conversión a escala de grises cuando sea posible
- Recorte automático de la región de interés (ROI)

## 🔒 Seguridad

### 1. Protección de Datos
- Los embeddings se almacenan de forma segura en la base de datos
- Las imágenes originales no se almacenan a menos que se configure explícitamente
- Comunicación segura mediante HTTPS

### 2. Autenticación y Autorización
- Uso de JWT para autenticación
- Control de acceso basado en roles (RBAC)
- Tiempo de expiración para tokens

### 3. Prevención de Ataques
- Validación de entrada estricta
- Protección contra inyección SQL
- Límites de tasa (rate limiting)
- Protección contra ataques de fuerza bruta

## 🚨 Manejo de Errores

### Códigos de Estado HTTP
- `200 OK`: Operación exitosa
- `400 Bad Request`: Error en los datos de entrada
- `401 Unauthorized`: No autenticado
- `403 Forbidden`: No autorizado
- `404 Not Found`: Recurso no encontrado
- `422 Unprocessable Entity`: Error de validación
- `500 Internal Server Error`: Error del servidor

### Mensajes de Error
```json
{
    "status": "error",
    "error": {
        "code": "FACE_NOT_FOUND",
        "message": "No se detectó ningún rostro en la imagen",
        "details": {
            "confidence_threshold": 0.5,
            "image_size": "800x600"
        }
    }
}
```

## 🧪 Pruebas

### 1. Pruebas Unitarias
- Pruebas de funciones individuales
- Mock de dependencias externas
- Cobertura de código > 80%

### 2. Pruebas de Integración
- Pruebas de extremo a extremo (E2E)
- Pruebas de rendimiento
- Pruebas de carga

### 3. Pruebas Manuales
- Verificación de la interfaz de usuario
- Pruebas de usabilidad
- Pruebas de compatibilidad

## 🚀 Despliegue

### 1. Requisitos del Servidor
- Sistema operativo: Ubuntu 20.04 LTS
- Docker y Docker Compose
- Nginx como proxy inverso
- Certificado SSL (Let's Encrypt)

### 2. Despliegue con Docker
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build: .
    restart: always
    env_file: .env.prod
    ports:
      - "8000:8000"
    depends_on:
      - db
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G

  db:
    image: postgres:13
    restart: always
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 1G

volumes:
  postgres_data:
```

### 3. Configuración de Nginx
```nginx
server {
    listen 80;
    server_name api.tudominio.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.tudominio.com;

    ssl_certificate /etc/letsencrypt/live/api.tudominio.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.tudominio.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 🔄 Mantenimiento

### 1. Monitoreo
- Uso de herramientas como Prometheus y Grafana
- Alertas para problemas críticos
- Registros detallados (logs)

### 2. Copias de Seguridad
- Copias de seguridad diarias de la base de datos
- Almacenamiento seguro de copias de seguridad
- Pruebas de recuperación periódicas

### 3. Actualizaciones
- Actualizaciones de seguridad regulares
- Migraciones de base de datos controladas
- Pruebas exhaustivas antes de implementar cambios

## 🐛 Solución de Problemas

### 1. Problemas Comunes

#### 1.1. No se detectan rostros
- Verifica que la imagen sea clara y el rostro esté bien iluminado
- Asegúrate de que el rostro ocupe al menos el 20% de la imagen
- Verifica que no haya objetos que obstruyan el rostro

#### 1.2. Bajo rendimiento
- Verifica los recursos del servidor (CPU, memoria)
- Reduce el tamaño de las imágenes antes de procesarlas
- Considera usar una GPU para acelerar el procesamiento

#### 1.3. Errores de conexión a la base de datos
- Verifica que PostgreSQL esté en ejecución
- Comprueba las credenciales en el archivo .env
- Asegúrate de que el puerto esté abierto en el firewall

### 2. Registros (Logs)
Los registros detallados están disponibles en:
- `/var/log/api/error.log`
- `/var/log/api/access.log`

### 3. Obtener Ayuda
Si necesitas ayuda adicional, por favor:
1. Revisa los registros de errores
2. Verifica la documentación
3. Abre un issue en el repositorio

## ❓ Preguntas Frecuentes

### ¿Qué formato de imagen es compatible?
La API admite imágenes en formato JPG, JPEG y PNG con un tamaño máximo de 5MB.

### ¿Cuál es la precisión del sistema?
La precisión depende de varios factores, pero en condiciones óptimas puede superar el 95%.

### ¿Cómo maneja la privacidad de los datos?
Los datos biométricos se almacenan de forma segura y se aplican medidas de seguridad avanzadas.

### ¿Es compatible con dispositivos móviles?
Sí, la API puede ser consumida desde cualquier dispositivo con conexión a Internet.

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Haz un fork del proyecto
2. Crea una rama para tu característica (`git checkout -b feature/AmazingFeature`)
3. Haz commit de tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Haz push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.

## 📞 Contacto

Para consultas técnicas o soporte, por favor contacta a:

- **Nombre del Proyecto**: API de Reconocimiento Facial
- **Desarrollador**: [Tu Nombre]
- **Correo Electrónico**: [tu@email.com]
- **Sitio Web**: [https://tusitio.com](https://tusitio.com)
- **Repositorio**: [https://github.com/tu-usuario/api-reconocimiento-facial](https://github.com/tu-usuario/api-reconocimiento-facial)

---

*Documentación generada el 22 de julio de 2023*
