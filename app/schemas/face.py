from pydantic import BaseModel
from typing import Optional

class RecognitionResult(BaseModel):
    name: str
    is_known: bool
    distance: Optional[float] = None

class FaceRegisterRequest(BaseModel):
    nombre: str

class FaceRegisterResponse(BaseModel):
    message: str
    name: str

class FaceDeleteResponse(BaseModel):
    message: str

class StatusResponse(BaseModel):
    status: str
    loaded_faces_count: int
    source: str
    database_connection: str
    message: str
