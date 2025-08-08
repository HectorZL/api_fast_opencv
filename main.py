from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
from typing import List, Dict, Any, Optional
import io
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Import database functions
from app.database import (
    save_face_embedding,
    get_face_embeddings,
    delete_face,
    load_known_faces_from_db
)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition and management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for known faces (cached from database)
known_faces: List[Dict[str, Any]] = []

def load_known_faces():
    """Load known faces from database into memory."""
    global known_faces
    known_faces = []
    try:
        db_faces = get_face_embeddings()
        for face_id, name, embedding in db_faces:
            known_faces.append({
                'id': face_id,
                'name': name,
                'encoding': np.array(embedding)
            })
        print(f"Loaded {len(known_faces)} known faces from database")
    except Exception as e:
        print(f"Error loading known faces: {e}")

@app.on_event("startup")
async def startup_event():
    """Load known faces when the application starts."""
    load_known_faces()

@app.post("/register")
async def register_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Register a new face with the system.
    
    - **name**: Name of the person
    - **file**: Image file containing the face
    """
    try:
        print(f"Starting face registration for: {name}")
        print(f"File received: {file.filename}, size: {file.size} bytes")
        
        # Read and process the image
        contents = await file.read()
        print(f"Read {len(contents)} bytes from the uploaded file")
        
        try:
            image = face_recognition.load_image_file(io.BytesIO(contents))
            print("Successfully loaded image")
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
        # Find all face encodings in the image
        print("Detecting faces in the image...")
        try:
            face_encodings = face_recognition.face_encodings(image)
            print(f"Found {len(face_encodings)} face(s) in the image")
        except Exception as e:
            print(f"Error detecting faces: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error detecting faces: {str(e)}")
        
        if not face_encodings:
            print("No faces found in the image")
            raise HTTPException(status_code=400, detail="No faces found in the image")
        
        # Use the first face found
        face_encoding = face_encodings[0]
        print("Successfully generated face encoding")
        
        # Save to database
        print("Saving face to database...")
        try:
            person_id = save_face_embedding(name, face_encoding.tolist())
            print(f"Successfully saved face with ID: {person_id}")
        except Exception as e:
            print(f"Database error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        
        # Update in-memory cache
        print("Updating in-memory face cache...")
        try:
            load_known_faces()
            print(f"Successfully updated in-memory cache. Total known faces: {len(known_faces)}")
        except Exception as e:
            print(f"Warning: Could not update in-memory cache: {str(e)}")
        
        return {
            "status": "success",
            "person_id": person_id,
            "name": name,
            "message": "Face registered successfully"
        }
        
    except HTTPException as he:
        # Re-raise HTTP exceptions as-is
        print(f"HTTP Exception: {he.detail}")
        raise he
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Unexpected error in register_face: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    Recognize faces in the uploaded image.
    
    - **file**: Image file containing faces to recognize
    """
    try:
        # Read and process the image
        contents = await file.read()
        image = face_recognition.load_image_file(io.BytesIO(contents))
        
        # Find all face encodings in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No faces found in the image")
        
        # Compare with known faces
        results = []
        for face_encoding, face_location in zip(face_encodings, face_locations):
            if not known_faces:
                matches = []
                face_distances = []
            else:
                # Get face distances to all known faces
                face_distances = face_recognition.face_distance(
                    [face['encoding'] for face in known_faces],
                    face_encoding
                )
                # Find best match
                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]
                
                # Use a threshold to determine a match (lower is more similar)
                face_match_threshold = 0.6
                matches = [best_match_distance <= face_match_threshold]
                
                if matches[0]:  # If we have a match
                    name = known_faces[best_match_index]['name']
                    person_id = known_faces[best_match_index]['id']
                    confidence = 1.0 - best_match_distance  # Convert to confidence score (0-1)
                else:
                    name = "Unknown"
                    person_id = None
                    confidence = 0.0
            
            top, right, bottom, left = face_location
            results.append({
                "person_id": person_id if 'person_id' in locals() else None,
                "name": name if 'name' in locals() else "Unknown",
                "confidence": float(confidence) if 'confidence' in locals() else 0.0,
                "location": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                }
            })
        
        return {
            "status": "success",
            "results": results,
            "total_faces": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/faces/{persona_id}")
async def delete_persona(persona_id: int):
    """
    Delete a person from the system (soft delete).
    
    - **persona_id**: ID of the person to delete
    """
    try:
        success = delete_face(persona_id)
        if success:
            # Update in-memory cache
            load_known_faces()
            return {
                "status": "success",
                "message": "Person deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Person not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/faces")
async def list_faces():
    """
    List all registered faces in the system.
    """
    try:
        return {
            "status": "success",
            "count": len(known_faces),
            "faces": [
                {
                    "id": face['id'],
                    "name": face['name']
                }
                for face in known_faces
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "known_faces_count": len(known_faces)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
