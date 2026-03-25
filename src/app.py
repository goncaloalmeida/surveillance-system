"""
Flask Web Application
Main web server for the surveillance system, providing web interface and API endpoints.
"""

from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import os
import json
import time
from datetime import datetime
from pathlib import Path
from src.logger import logger
from src.utils import load_config
from src.recognition import (
    load_known_faces, 
    start_recognition_workers, 
    get_face_events,
    known_face_names,
    identified_people_cache
)
from src.camera_processing import (
    initialize_yolo, 
    start_camera_threads, 
    get_camera_frame, 
    get_camera_stats
)

REFERENCE_FACES_DIR = Path('reference_faces')
DEFAULT_AVATAR_PATH = '/static/assets/default-avatar.svg'

# Load configuration
CONFIG = load_config()

# Initialize Flask application
app = Flask(__name__, 
            template_folder='../web/templates',
            static_folder='../web/static')

# Global variable to track initialization status
is_initialized = False


def resolve_reference_image_path(name):
    """Return the public asset path for a reference image when it exists."""
    for ext in ['.jpg', '.jpeg', '.png', '.webp']:
        reference_file = REFERENCE_FACES_DIR / f"{name}{ext}"
        if reference_file.exists():
            return f"/reference-images/{name}{ext}"
    return DEFAULT_AVATAR_PATH

def initialize():
    """
    Initialize the surveillance system.
    
    This function performs all necessary setup steps:
    - Sets up logging system
    - Creates required directories
    - Generates default assets (logo)
    - Loads the reference identity database
    - Initializes YOLO model
    - Starts recognition worker threads
    - Starts camera processing threads
    
    Returns:
        int: Number of reference identities loaded
    """
    global is_initialized
    
    if is_initialized:
        logger.info("System already initialized")
        return 0
    
    logger.info("Starting system initialization...")
    
    try:
        # Step 1: Setup logging system
        logger.setup(CONFIG)
        logger.info("Logging system configured")
        
        # Step 2: Create required directories
        assets_dir = os.path.join("web", "static", "assets")
        os.makedirs(assets_dir, exist_ok=True)
        logger.info(f"Created assets directory: {assets_dir}")
               
        # Step 4: Load reference identities from disk
        logger.info("Loading reference identity database...")
        face_count = load_known_faces()
        logger.info(f"Loaded {face_count} reference identities")
        
        # Step 5: Initialize YOLO model for person detection
        logger.info("Initializing YOLO model...")
        initialize_yolo()
        logger.info("YOLO model initialized")
        
        # Step 6: Start face recognition worker threads
        logger.info("Starting face recognition workers...")
        start_recognition_workers()
        
        # Step 7: Start camera processing threads
        logger.info("Starting camera threads...")
        start_camera_threads()
        
        is_initialized = True
        logger.info("System initialization completed successfully")
        
        return face_count
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        raise

def generate_frames(camera_id):
    """
    Generator function to stream camera frames as MJPEG.
    
    This function continuously yields frames from a specific camera,
    encoded as JPEG images in multipart format for HTTP streaming.
    
    Args:
        camera_id (str): Camera identifier
        
    Yields:
        bytes: JPEG-encoded frame with multipart headers
    """
    frame_count = 0
    
    while True:
        try:
            # Get the latest frame from the camera
            frame = get_camera_frame(camera_id)
            
            if frame is None:
                # If no frame available, wait and retry
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Encode frame as JPEG
            # Quality parameter (0-100): higher = better quality but larger file
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, CONFIG.get("jpeg_quality")]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if not ret:
                logger.warning(f"Failed to encode frame for camera {camera_id}")
                continue
            
            # Convert to bytes
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format for MJPEG streaming
            # This format is understood by browsers for continuous video streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            logger.error(f"Error generating frame for camera {camera_id}: {e}")
            time.sleep(0.5)

# ============================================================================
# Web Routes
# ============================================================================

@app.route('/')
def index():
    """
    Render the main dashboard page.
    
    Returns:
        HTML: Rendered index page showing all cameras
    """
    try:
        cameras = CONFIG.get("cameras", [])
        return render_template('index.html', cameras=cameras)
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return f"Error: {e}", 500

@app.route('/camera')
def camera():
    """
    Render the camera detail page.
    
    Returns:
        HTML: Rendered camera page with detailed view
    """
    try:
        cameras = CONFIG.get("cameras", [])
        return render_template('camera.html', cameras=cameras)
    except Exception as e:
        logger.error(f"Error rendering camera page: {e}")
        return f"Error: {e}", 500

# ============================================================================
# API Routes
# ============================================================================

@app.route('/api/people/detected')
def api_people_detected():
    """
    Get only people who have been detected today with their cooldown status.
    
    Returns:
        JSON: List of detected people with name, photo_path, status, in_cooldown
    """
    try:
        from src.recognition import people_cooldown, COOLDOWN_MINUTES
        import time
        
        # Get recent events to determine who was detected
        events_list = get_face_events()
        detected_people = {}
        
        if events_list and isinstance(events_list, list):
            for event in events_list:
                if isinstance(event, dict):
                    person_name = event.get('person', '')
                    event_type = event.get('event_type', 'entry')
                    timestamp = event.get('timestamp', 0)
                    
                    if person_name:
                        # Keep most recent event per person
                        if person_name not in detected_people or timestamp > detected_people[person_name]['timestamp']:
                            detected_people[person_name] = {
                                'status': 'inside' if event_type == 'entry' else 'outside',
                                'timestamp': timestamp
                            }
        
        # Build response with cooldown info
        people = []
        current_time = time.time()
        cooldown_seconds = COOLDOWN_MINUTES * 60
        
        for name in detected_people.keys():
            photo_path = resolve_reference_image_path(name)
            
            # Check if person was recently detected (still in cooldown = present)
            # in_cooldown TRUE means: recently detected, show in COLOR
            # in_cooldown FALSE means: detection expired, show in GRAYSCALE
            in_cooldown = False
            cooldown_remaining = 0
            if name in people_cooldown:
                time_since_detection = current_time - people_cooldown[name]
                if time_since_detection < cooldown_seconds:
                    in_cooldown = True  # Recently detected = still present
                    cooldown_remaining = int(cooldown_seconds - time_since_detection)
            
            people.append({
                'id': name.lower().replace(' ', '_'),
                'name': name,
                'photo_path': photo_path,
                'status': detected_people[name]['status'],
                'in_cooldown': in_cooldown,
                'cooldown_remaining': cooldown_remaining,
                'last_seen': detected_people[name]['timestamp']
            })
        
        # Sort by most recent first
        people.sort(key=lambda x: x['last_seen'], reverse=True)
        
        return jsonify(people)
    except Exception as e:
        logger.error(f"Error in /api/people/detected: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/people')
def api_people():
    """
    Get list of all known people in the database with their current status.
    
    Returns:
        JSON: List of known people with name, photo_path, and status (inside/outside)
    """
    try:
        # Get recent events to determine who is inside
        events_list = get_face_events()  # Returns a list directly
        people_status = {}  # name -> 'inside' or 'outside'
        
        # Process events to determine current status
        if events_list and isinstance(events_list, list):
            for event in events_list:
                if isinstance(event, dict):
                    person_name = event.get('person', '')
                    event_type = event.get('event_type', 'entry')
                    # Most recent event determines status
                    if person_name and person_name not in people_status:
                        people_status[person_name] = 'inside' if event_type == 'entry' else 'outside'
        
        # Build the response with optional reference images.
        people = []
        for name in known_face_names:
            people.append({
                'id': name.lower().replace(' ', '_'),
                'name': name,
                'photo_path': resolve_reference_image_path(name),
                'status': people_status.get(name, 'outside')
            })
        
        return jsonify(people)
    except Exception as e:
        logger.error(f"Error in /api/people: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/events')
def api_events():
    """
    Get recent recognition events.
    
    Returns:
        JSON: List of recent face recognition events with timestamps and details
    """
    try:
        events = get_face_events()
        logger.debug(f"API: Returning {len(events)} recent events")
        return jsonify(events)
    except Exception as e:
        logger.error(f"Error in /api/events: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cameras/stats')
def api_camera_stats():
    """
    Get statistics for all cameras.
    
    Returns:
        JSON: Dictionary with camera IDs as keys and their statistics as values
              Including: status, fps, persons_detected, last_detection, etc.
    """
    try:
        stats = get_camera_stats()  # Get stats for all cameras
        
        # Convert timestamp to readable format
        for camera_id, camera_stat in stats.items():
            if camera_stat.get("last_detection"):
                # Convert Unix timestamp to ISO format string
                camera_stat["last_detection_time"] = datetime.fromtimestamp(
                    camera_stat["last_detection"]
                ).isoformat()
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in /api/cameras/stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cameras/config')
def api_cameras_config():
    """
    Get camera configuration from config file.
    
    Returns:
        JSON: Dictionary with cameras list containing id, name, enabled status
    """
    try:
        cameras = CONFIG.get("cameras", [])
        return jsonify({"cameras": cameras})
    except Exception as e:
        logger.error(f"Error in /api/cameras/config: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cameras/detections')
def api_cameras_detections():
    """
    Get detection counters for each camera (identified and unidentified persons).
    
    Returns:
        JSON: List of cameras with detection counters
                            [{"id": "camera1", "name": "Lobby Entrance", 
                "identified": 5, "unidentified": 2, "total": 7}]
    """
    try:
        from src.recognition import people_cooldown, COOLDOWN_MINUTES
        import time
        
        cameras = CONFIG.get("cameras", [])
        enabled_cameras = [c for c in cameras if c.get("enabled", False)]
        
        camera_detections = []
        current_time = time.time()
        cooldown_seconds = COOLDOWN_MINUTES * 60
        
        # Get face events to count identified people per camera
        events_list = get_face_events()
        
        for camera in enabled_cameras:
            camera_id = camera.get("id")
            camera_name = camera.get("name", camera_id)
            
            # Count identified people for this camera (only those still in cooldown = recently detected)
            identified_count = 0
            identified_people = set()
            
            if events_list and isinstance(events_list, list):
                for event in events_list:
                    if isinstance(event, dict):
                        event_camera = event.get('camera', '')
                        person_name = event.get('person', '')
                        
                        # Only count if it's from this camera and person is still in cooldown
                        if event_camera == camera_name and person_name:
                            if person_name in people_cooldown:
                                time_since_detection = current_time - people_cooldown[person_name]
                                if time_since_detection < cooldown_seconds:
                                    identified_people.add(person_name)
            
            identified_count = len(identified_people)
            
            # Get camera stats for total detections
            stats = get_camera_stats(camera_id)
            total_detected = stats.get("persons_detected", 0)
            
            # Unidentified = total detected - identified
            # (persons detected by YOLO but not recognized)
            unidentified_count = max(0, total_detected - identified_count)
            
            camera_detections.append({
                "id": camera_id,
                "name": camera_name,
                "identified": identified_count,
                "unidentified": unidentified_count,
                "total": total_detected,
                "status": stats.get("status", "unknown"),
                "fps": round(stats.get("fps", 0), 1)
            })
        
        logger.debug(f"API: Returning detections for {len(camera_detections)} cameras")
        return jsonify(camera_detections)
    except Exception as e:
        logger.error(f"Error in /api/cameras/detections: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ============================================================================
# Streaming Routes
# ============================================================================

@app.route('/camera/stream/<camera_id>')
def camera_stream(camera_id):
    """
    Stream live video from a specific camera.
    
    Args:
        camera_id (str): Camera identifier from URL
        
    Returns:
        Response: MJPEG stream response with multipart content type
    """
    try:
        logger.info(f"Starting video stream for camera {camera_id}")
        
        # Return MJPEG stream response
        # mimetype specifies multipart format with boundary marker
        return Response(
            generate_frames(camera_id),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Error streaming camera {camera_id}: {e}")
        return f"Error: {e}", 500

@app.route('/reference-images/<filename>')
def serve_reference_image(filename):
    """
    Serve reference images from the public demo directory.
    
    Args:
        filename (str): Name of the reference image file
        
    Returns:
        File: The requested reference image file
    """
    try:
        reference_dir = os.path.abspath("reference_faces")
        logger.debug(f"Serving reference image: {filename}")
        return send_from_directory(reference_dir, filename)
    except Exception as e:
        logger.error(f"Error serving reference image {filename}: {e}")
        return "Image not found", 404