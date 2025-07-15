import os
import cv2
import numpy as np
import face_recognition
from pathlib import Path
from dotenv import load_dotenv
import logging
import json
from datetime import datetime

# Import our new utility functions
from image_utils import load_image_with_retry, check_image_quality

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('photo_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_directories(single_image_path=None):
    """Set up and verify required directories or process a single image."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get base directory
        base_dir = Path(__file__).resolve().parent
        logger.info(f"Base directory: {base_dir}")
        
        # If a single image path is provided, use it directly
        if single_image_path:
            single_image_path = Path(single_image_path)
            if not single_image_path.is_absolute():
                single_image_path = base_dir / single_image_path
                
            if not single_image_path.exists():
                logger.error(f"Image file not found: {single_image_path}")
                return base_dir, None, None
                
            logger.info(f"Processing single image: {single_image_path}")
            return base_dir, single_image_path, base_dir / "analysis_results"
        
        # Otherwise, use the default reference_photos directory
        image_folder = base_dir / "reference_photos"
        logger.info(f"Looking for images in: {image_folder}")
        
        # Debug: List all files and directories in base directory
        logger.info("\nContents of base directory:")
        for i, item in enumerate(base_dir.iterdir(), 1):
            item_type = "DIR " if item.is_dir() else "FILE"
            logger.info(f"  {i:2d}. {item_type} - {item.name}")
        
        # Create results directory
        results_dir = base_dir / "analysis_results"
        results_dir.mkdir(exist_ok=True)
        logger.info(f"Results will be saved to: {results_dir}")
        
        # Verify image directory exists
        if not image_folder.exists():
            # Try to find similar directories
            similar_dirs = [d for d in base_dir.iterdir() 
                          if d.is_dir() and 'reference' in d.name.lower()]
            
            error_msg = [
                f"Error: Directory '{image_folder.name}' not found in {base_dir}",
                f"Current working directory: {Path.cwd()}",
                f"Directory exists: {image_folder.exists()}",
                f"Parent exists: {image_folder.parent.exists()}",
                f"Is absolute path: {image_folder.is_absolute()}",
                f"Resolved path: {image_folder.resolve()}"
            ]
            
            if similar_dirs:
                error_msg.append("\nDid you mean one of these directories?")
                for i, d in enumerate(similar_dirs, 1):
                    error_msg.append(f"  {i}. {d}")
            
            error_msg = "\n".join(error_msg)
            logger.error(error_msg)
            print("\n" + "!"*80)
            print("ERROR: Directory not found. Please check the following information:")
            print("-"*80)
            print(f"Looking for: {image_folder}")
            print(f"Current working directory: {Path.cwd()}")
            print(f"Directory exists: {image_folder.exists()}")
            print(f"Parent exists: {image_folder.parent.exists()}")
            print("\nAvailable directories:")
            for i, item in enumerate(base_dir.iterdir(), 1):
                if item.is_dir():
                    print(f"  {i}. {item}")
            print("!"*80 + "\n")
            
            # Ask user to confirm the correct directory
            while True:
                print("\nPlease enter the correct directory name (or press Enter to create it):")
                user_input = input("Directory name: ").strip()
                
                if not user_input:
                    try:
                        image_folder.mkdir(exist_ok=True)
                        print(f"\nCreated directory: {image_folder}")
                        print("Please add your images to this directory and run the script again.")
                        exit(0)
                    except Exception as e:
                        print(f"\nError creating directory: {e}")
                        continue
                
                # Check if the user provided a full path or relative path
                user_path = Path(user_input)
                if user_path.is_absolute():
                    if user_path.exists() and user_path.is_dir():
                        image_folder = user_path
                        break
                else:
                    # Try as relative path
                    test_path = base_dir / user_path
                    if test_path.exists() and test_path.is_dir():
                        image_folder = test_path
                        break
                    
                    # Try to find a matching directory
                    matches = list(base_dir.glob(f"*{user_path}*"))
                    if len(matches) == 1 and matches[0].is_dir():
                        image_folder = matches[0]
                        print(f"Found matching directory: {image_folder}")
                        break
                    elif len(matches) > 1:
                        print("\nMultiple matches found. Please be more specific:")
                        for i, match in enumerate(matches, 1):
                            print(f"  {i}. {match}")
                        continue
                
                print(f"Directory not found: {user_input}")
        
        logger.info(f"Using image directory: {image_folder}")
        return base_dir, image_folder, results_dir
    
    except Exception as e:
        logger.exception("Error in setup_directories():")
        raise

def find_image_files(image_folder: Path) -> list:
    """Find all valid image files in the specified directory."""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.jfif', '.webp']
    image_files = [f for f in image_folder.glob('*') 
                  if f.suffix.lower() in valid_extensions]
    
    if not image_files:
        error_msg = f"No se encontraron archivos de imagen {', '.join(valid_extensions)} en el directorio."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    return sorted(image_files)

def process_face_detection(image: np.ndarray, min_face_size: int = 30) -> dict:
    """Process face detection on the given image."""
    result = {
        'face_count': 0,
        'face_locations': [],
        'face_encodings': [],
        'face_sizes': [],
        'errors': []
    }
    
    try:
        # Convert to RGB if needed (face_recognition uses RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image
            if np.array_equal(image[0, 0], image[0, 0, ::-1]):  # Check if already RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        result['face_count'] = len(face_locations)
        
        if not face_locations:
            result['errors'].append("No faces detected")
            return result
        
        # Process each face
        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_size = (right - left) * (bottom - top)
            
            if face_size < min_face_size * min_face_size:
                result['errors'].append(f"Face too small: {face_size}px²")
                continue
                
            result['face_sizes'].append(face_size)
            result['face_locations'].append(face_location)
            
            # Get face encodings
            try:
                encodings = face_recognition.face_encodings(
                    rgb_image, 
                    [face_location],
                    model="large"  # Use the larger model for better accuracy
                )
                if encodings:
                    result['face_encodings'].append(encodings[0].tolist())
            except Exception as e:
                error_msg = f"Face encoding failed: {str(e)}"
                logger.warning(error_msg)
                result['errors'].append(error_msg)
    
    except Exception as e:
        error_msg = f"Face detection failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result['errors'].append(error_msg)
    
    return result

def analyze_image(image_path: Path) -> dict:
    """Analyze a single image with detailed error handling."""
    result = {
        'file_name': image_path.name,
        'file_path': str(image_path.absolute()),
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'warnings': [],
        'errors': []
    }
    
    logger.info(f"\n{'='*80}\nProcessing: {image_path.name}")
    
    # Step 1: Load the image with retry
    image, load_info = load_image_with_retry(
        image_path,
        max_attempts=3,
        initial_delay=0.5,
        backoff_factor=2.0,
        convert_to_rgb=True
    )
    
    # Update result with loading info
    result.update({
        'load_info': load_info,
        'success': load_info['success']
    })
    
    if not load_info['success']:
        error_msg = f"Failed to load image after {load_info['attempts']} attempts"
        logger.error(error_msg)
        result['errors'].extend(load_info['errors'])
        return result
    
    # Step 2: Check image quality
    quality_info = check_image_quality(image)
    result['quality'] = quality_info
    
    if quality_info['issues']:
        logger.warning(f"Quality issues: {', '.join(quality_info['issues'])}")
        result['warnings'].extend(quality_info['issues'])
    
    # Step 3: Process face detection
    face_info = process_face_detection(image)
    result['face_info'] = face_info
    
    if 'face_count' in face_info and face_info['face_count'] > 0:
        logger.info(f"Detected {face_info['face_count']} face(s) in the image")
        if 'face_sizes' in face_info and face_info['face_sizes']:
            logger.info(f"Face sizes: {', '.join(map(str, face_info['face_sizes']))}px²")
    else:
        warning_msg = "No faces detected in the image"
        logger.warning(warning_msg)
        result['warnings'].append(warning_msg)
    
    # Add success flag
    result['success'] = True
    
    return result

def save_results(results: list, output_dir: Path):
    """Save analysis results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"photo_analysis_{timestamp}.json"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'total_images': len(results),
                'successful_analyses': sum(1 for r in results if r.get('success', False)),
                'images': results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        return output_file
    except Exception as e:
        error_msg = f"Failed to save results: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise

def main():
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='Analyze photos for face detection')
        parser.add_argument('--image', '-i', help='Process a single image file')
        parser.add_argument('--directory', '-d', help='Process all images in a directory')
        args = parser.parse_args()
        
        # Check if no arguments were provided
        if not args.image and not args.directory:
            print("Error: You must specify either --image or --directory")
            parser.print_help()
            return 1
            
        # Setup based on input type
        if args.image:
            # Single image mode
            base_dir, image_path, results_dir = setup_directories(args.image)
            if not image_path or not image_path.is_file():
                logger.error(f"Image file not found: {args.image}")
                return 1
                
            if not results_dir:
                results_dir = base_dir / "analysis_results"
                results_dir.mkdir(exist_ok=True)
                
            result = analyze_image(image_path)
            output_file = save_results([result], results_dir)
            print(f"\nAnalysis complete! Results saved to: {output_file}")
            return 0 if result.get('success', False) else 1
            
        elif args.directory:
            # Directory mode
            input_dir = Path(args.directory)
            if not input_dir.is_dir():
                logger.error(f"Directory not found: {args.directory}")
                return 1
                
            # Create results directory with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = input_dir.parent / f"analysis_results_{timestamp}"
            results_dir.mkdir(exist_ok=True)
            
            # Find all image files
            image_files = find_image_files(input_dir)
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        
        if not image_files:
            logger.warning("No image files found in the specified directory")
            return 1
            
        # Process each image
        results = []
        for i, img_path in enumerate(image_files, 1):
            try:
                logger.info(f"\nProcessing image {i}/{len(image_files)}: {img_path.name}")
                result = analyze_image(img_path)
                results.append(result)
                logger.info(f"Completed: {img_path.name} - {'Success' if result.get('success') else 'Failed'}")
            except Exception as e:
                error_msg = f"Error processing {img_path.name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                results.append({
                    'file_name': img_path.name,
                    'file_path': str(img_path.absolute()),
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    'errors': [error_msg]
                })
        
        # Save results
        output_file = save_results(results, results_dir)
        
        # Print summary
        successful = sum(1 for r in results if r.get('success', False))
        logger.info(f"\n{'='*40}\n"
                   f"Analysis complete!\n"
                   f"Total images: {len(results)}\n"
                   f"Successfully processed: {successful}\n"
                   f"Failed: {len(results) - successful}\n"
                   f"Results saved to: {output_file}\n"
                   f"{'='*40}")
        
        return 0
    
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
