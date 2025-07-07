import os
import time
import asyncio
import logging
from typing import List, Set
import psycopg2
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PHOTOS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "reference_photos")
CLEANUP_INTERVAL = 15  # seconds

# Ensure photos directory exists
os.makedirs(PHOTOS_DIR, exist_ok=True)

def get_db_connection():
    """Helper to get a database connection."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "mi_base_de_datos"),
            user=os.getenv("DB_USER", "mi_usuario"),
            password=os.getenv("DB_PASSWORD", "mi_contraseña_segura")
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def get_registered_names() -> Set[str]:
    """Get a set of all registered names from the database."""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            return set()
            
        with conn.cursor() as cur:
            cur.execute("SELECT nombre FROM mis_personas")
            return {row[0] for row in cur.fetchall()}
    except Exception as e:
        logger.error(f"Error fetching registered names: {e}")
        return set()
    finally:
        if conn:
            conn.close()

def cleanup_orphaned_photos():
    """Delete photos that don't have a corresponding database entry."""
    try:
        # Get all registered names
        registered_names = get_registered_names()
        
        # Get all photo files in the directory
        photo_files = set()
        for filename in os.listdir(PHOTOS_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Remove extension and add to set
                name = os.path.splitext(filename)[0]
                photo_files.add((name, filename))
        
        # Find and delete orphaned photos
        deleted_count = 0
        for name, filename in photo_files:
            if name not in registered_names:
                try:
                    filepath = os.path.join(PHOTOS_DIR, filename)
                    os.remove(filepath)
                    logger.info(f"Deleted orphaned photo: {filename}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting photo {filename}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleanup complete. Deleted {deleted_count} orphaned photos.")
        
    except Exception as e:
        logger.error(f"Error during photo cleanup: {e}")

async def run_photo_cleanup():
    """Run the photo cleanup in a loop with the specified interval."""
    logger.info("Starting photo cleanup service...")
    while True:
        try:
            cleanup_orphaned_photos()
        except Exception as e:
            logger.error(f"Error in photo cleanup task: {e}")
        
        # Wait for the next cleanup cycle
        await asyncio.sleep(CLEANUP_INTERVAL)
