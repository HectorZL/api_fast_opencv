"""
Utility functions for robust image loading and processing.
"""
import os
import cv2
import time
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('image_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def load_image_with_retry(
    image_path: Union[str, Path],
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    convert_to_rgb: bool = True,
    **kwargs
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Load an image with retry mechanism and detailed error reporting.

    Args:
        image_path: Path to the image file
        max_attempts: Maximum number of loading attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay between retries
        convert_to_rgb: Whether to convert BGR to RGB
        **kwargs: Additional arguments for cv2.imread()

    Returns:
        Tuple of (image, info_dict) where:
        - image: Loaded image array or None if all attempts failed
        - info_dict: Dictionary with debugging information and error details
    """
    image_path = Path(image_path) if not isinstance(image_path, Path) else image_path
    info = {
        'success': False,
        'attempts': 0,
        'errors': [],
        'file_size': None,
        'file_permissions': None,
        'file_exists': None,
        'file_path': str(image_path.absolute()),
        'file_extension': image_path.suffix.lower(),
        'image_shape': None,
        'image_dtype': None,
        'processing_time': None
    }

    # Check file existence and permissions first
    if not image_path.exists():
        error_msg = f"File does not exist: {image_path}"
        logger.error(error_msg)
        info['errors'].append(error_msg)
        return None, info
    
    info['file_exists'] = True
    info['file_size'] = os.path.getsize(image_path)
    
    try:
        info['file_permissions'] = oct(os.stat(image_path).st_mode)[-3:]
    except Exception as e:
        info['file_permissions'] = f"Error: {str(e)}"

    # Check file size
    if info['file_size'] == 0:
        error_msg = f"File is empty: {image_path}"
        logger.error(error_msg)
        info['errors'].append(error_msg)
        return None, info

    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        info['attempts'] = attempt
        start_time = time.time()
        
        try:
            # Try to read the image
            image = cv2.imread(str(image_path), **kwargs)
            
            if image is None:
                error_msg = f"Attempt {attempt}: Failed to decode image (cv2.imread returned None)"
                logger.warning(error_msg)
                info['errors'].append(error_msg)
                
                # Try alternative reading method on last attempt
                if attempt == max_attempts:
                    try:
                        with open(image_path, 'rb') as f:
                            image_data = np.frombuffer(f.read(), np.uint8)
                        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                        if image is not None:
                            logger.info("Successfully read image using alternative method")
                    except Exception as e:
                        error_msg = f"Alternative read method failed: {str(e)}"
                        logger.error(error_msg)
                        info['errors'].append(error_msg)
            
            # If we have a valid image
            if image is not None:
                info['success'] = True
                info['image_shape'] = image.shape
                info['image_dtype'] = str(image.dtype)
                
                if convert_to_rgb and len(image.shape) >= 3:  # Only convert if it's a color image
                    try:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        info['color_space'] = 'RGB'
                    except Exception as e:
                        error_msg = f"Color conversion failed: {str(e)}"
                        logger.warning(error_msg)
                        info['errors'].append(error_msg)
                
                info['processing_time'] = time.time() - start_time
                logger.info(f"Successfully loaded image: {image_path}")
                return image, info
            
        except Exception as e:
            error_msg = f"Attempt {attempt}: Unexpected error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            info['errors'].append(error_msg)
        
        # If we get here, the attempt failed
        if attempt < max_attempts:
            logger.info(f"Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
            delay *= backoff_factor
    
    # If all attempts failed
    info['processing_time'] = time.time() - start_time if 'start_time' in locals() else None
    logger.error(f"Failed to load image after {max_attempts} attempts: {image_path}")
    return None, info

def check_image_quality(
    image: np.ndarray,
    min_width: int = 50,
    min_height: int = 50,
    min_face_size: int = 30
) -> Dict[str, Any]:
    """
    Check image quality and detect potential issues.
    
    Args:
        image: Input image
        min_width: Minimum acceptable width in pixels
        min_height: Minimum acceptable height in pixels
        min_face_size: Minimum face size to consider valid
        
    Returns:
        Dictionary with quality metrics and issues
    """
    result = {
        'width': image.shape[1],
        'height': image.shape[0],
        'channels': image.shape[2] if len(image.shape) > 2 else 1,
        'is_too_small': False,
        'is_too_large': False,
        'is_low_contrast': False,
        'is_blurry': False,
        'has_faces': False,
        'face_count': 0,
        'face_sizes': [],
        'issues': []
    }
    
    # Check image dimensions
    if result['width'] < min_width or result['height'] < min_height:
        result['is_too_small'] = True
        result['issues'].append(f"Image is too small ({result['width']}x{result['height']})")
    
    # Check if image is too large (optional)
    if result['width'] > 8000 or result['height'] > 8000:
        result['is_too_large'] = True
        result['issues'].append(f"Image is very large ({result['width']}x{result['height']})")
    
    # Convert to grayscale for quality checks
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Check contrast (using standard deviation of pixel intensities)
    contrast = np.std(gray)
    if contrast < 20:  # Threshold for low contrast
        result['is_low_contrast'] = True
        result['issues'].append(f"Low contrast (std dev: {contrast:.1f})")
    
    # Check blur (using Laplacian variance)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < 50:  # Threshold for blurry image
        result['is_blurry'] = True
        result['issues'].append(f"Image is blurry (Laplacian variance: {blur:.1f})")
    
    # Try to detect faces if the image is large enough
    if result['width'] >= min_face_size and result['height'] >= min_face_size:
        try:
            # Use a more efficient face detector for this check
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size)
            )
            
            result['face_count'] = len(faces)
            result['has_faces'] = len(faces) > 0
            result['face_sizes'] = [w * h for (x, y, w, h) in faces]
            
            if not result['has_faces']:
                result['issues'].append("No faces detected")
            
        except Exception as e:
            logger.warning(f"Face detection failed: {str(e)}")
            result['issues'].append("Face detection failed")
    
    return result
