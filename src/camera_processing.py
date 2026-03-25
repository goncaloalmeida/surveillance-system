"""
Camera Processing Module
Handles YOLO object detection, camera streaming, and frame processing for the surveillance system.
"""

import cv2
import threading
import time
import queue
from collections import deque
from ultralytics import YOLO
from src.logger import logger
from src.utils import load_config
from src.recognition import recognition_queue  # Import shared queue for face recognition

# Global configuration
CONFIG = load_config()

# Global YOLO model instance (lazy loaded)
yolo_model = None
yolo_lock = threading.Lock()

# Global dictionaries to store camera data
camera_threads = {}
camera_frames = {}  # Stores latest frame for each camera
camera_stats = {}   # Stores detection statistics for each camera

# Thread control flags
stop_threads = False

def initialize_yolo():
    """
    Initialize the YOLO model with lazy loading.
    This function is thread-safe and ensures the model is loaded only once.
    
    Returns:
        YOLO: The loaded YOLO model instance
    """
    global yolo_model
    
    # Use lock to prevent multiple threads from loading the model simultaneously
    with yolo_lock:
        if yolo_model is None:
            try:
                model_path = CONFIG.get("yolo_model")
                logger.info(f"Loading YOLO model from {model_path}")
                yolo_model = YOLO(model_path)
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                raise
    
    return yolo_model

def detect_persons(frame):
    """
    Detect persons in a frame using YOLO model.
    
    Args:
        frame: OpenCV image frame (numpy array)
        
    Returns:
        list: List of detection dictionaries containing bounding box coordinates,
              confidence scores, and class information
    """
    global yolo_model
    
    # Lazy load YOLO model if not already loaded
    if yolo_model is None:
        initialize_yolo()
    
    detections = []
    
    try:
        # Get YOLO configuration
        yolo_config = CONFIG.get("performance").get("yolo")
        conf_threshold = yolo_config.get("conf")
        max_det = yolo_config.get("max_det")
        imgsz = yolo_config.get("imgsz")
        
        # Run YOLO inference on the frame with all parameters
        # verbose=False suppresses console output
        results = yolo_model(frame, conf=conf_threshold, max_det=max_det, imgsz=imgsz, verbose=False)
        
        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract class ID (0 = person in COCO dataset)
                class_id = int(box.cls[0])
                
                # Only process person detections (class_id == 0)
                if class_id == 0:
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    # Only include detections above confidence threshold
                    min_confidence = CONFIG.get("performance").get("yolo").get("conf")
                    
                    # Log all detections for debugging (even below threshold)
                    if class_id == 0:  # Person class
                        logger.debug(f"YOLO detected person: conf={confidence:.2f}, threshold={min_confidence}")
                    
                    if confidence >= min_confidence:
                        detections.append({
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "confidence": confidence,
                            "class_id": class_id
                        })
        
    except Exception as e:
        logger.error(f"Error during person detection: {e}")
    
    return detections

def add_face_to_queue(face_data):
    """
    Add a detected face to the recognition queue for processing.
    
    Args:
        face_data (dict): Dictionary containing face image, camera_id, timestamp, and bbox
        
    Returns:
        bool: True if face was added to queue, False if queue is full
    """
    try:
        logger.debug(f"Adding face to queue (qsize: {recognition_queue.qsize()})")
        # Try to add to queue without blocking (raises queue.Full if full)
        recognition_queue.put_nowait(face_data)
        logger.debug(f"Face added to queue (qsize: {recognition_queue.qsize()})")
        return True
    except queue.Full:
        logger.warning("Face recognition queue is full, skipping face")
        return False
    except Exception as e:
        logger.error(f"Error adding item to recognition queue: {e}", exc_info=True)
        return False

def process_camera_stream(camera_id, camera_config):
    """
    Process video stream from a single camera.
    This function runs in a separate thread for each camera.
    
    Args:
        camera_id (str): Unique identifier for the camera
        camera_config (dict): Camera configuration containing ip, username, password
    """
    global stop_threads, camera_frames, camera_stats
    
    # Build RTSP URL from config
    ip = camera_config.get("ip")
    username = camera_config.get("username")
    password = camera_config.get("password")
    rtsp_url = f"rtsp://{username}:{password}@{ip}:554/stream1"
    
    camera_name = camera_config.get("name", camera_id)
    frame_skip = CONFIG.get("performance").get("frame_skip")
    
    logger.info(f"Starting camera: {camera_name} ({camera_id})")
    
    # Initialize camera statistics
    camera_stats[camera_id] = {
        "status": "connecting",
        "fps": 0,
        "persons_detected": 0,
        "last_detection": None,
        "frame_count": 0,
        "error_count": 0
    }
    
    # Open video capture
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera stream: {camera_id}")
        camera_stats[camera_id]["status"] = "error"
        return
    
    logger.info(f"Camera {camera_id} connected successfully")
    camera_stats[camera_id]["status"] = "active"
    
    frame_counter = 0
    fps_counter = deque(maxlen=30)  # Store last 30 frame timestamps for FPS calculation
    
    logger.info(f"{camera_id}: Starting frame processing loop, frame_skip={frame_skip}")
    
    try:
        while not stop_threads:
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame from {camera_id}")
                camera_stats[camera_id]["error_count"] += 1
                
                # Try to reconnect after multiple failures
                if camera_stats[camera_id]["error_count"] > 10:
                    logger.info(f"Attempting to reconnect camera {camera_id}")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(rtsp_url)
                    camera_stats[camera_id]["error_count"] = 0
                
                time.sleep(0.1)
                continue
            
            # Reset error count on successful frame read
            camera_stats[camera_id]["error_count"] = 0
            camera_stats[camera_id]["frame_count"] += 1
            
            # Increment frame counter FIRST (before any checks)
            frame_counter += 1
            
            # Store latest frame for streaming
            camera_frames[camera_id] = frame.copy()
            
            # Calculate FPS
            current_time = time.time()
            fps_counter.append(current_time)
            if len(fps_counter) > 1:
                time_diff = fps_counter[-1] - fps_counter[0]
                camera_stats[camera_id]["fps"] = len(fps_counter) / time_diff if time_diff > 0 else 0
            
            # Log frame processing every 100 frames
            if frame_counter % 100 == 0:
                logger.debug(f"{camera_id}: Processed {frame_counter} frames")
            
            # Process only every Nth frame to reduce CPU usage
            try:
                if frame_counter % frame_skip == 0:
                    # Detect persons in the frame
                    detections = detect_persons(frame)
                    
                    if detections:
                        camera_stats[camera_id]["persons_detected"] = len(detections)
                        camera_stats[camera_id]["last_detection"] = time.time()
                        
                        # Only log at DEBUG level to avoid spam
                        logger.debug(f"{camera_id}: YOLO detected {len(detections)} person(s)")
                        
                        # Extract faces from detections (no drawing needed)
                        faces_sent = 0
                        for detection in detections:
                            x1, y1, x2, y2 = detection["x1"], detection["y1"], detection["x2"], detection["y2"]
                            
                            # Extract face region for recognition (upper portion of detected person)
                            # Assume face is in the top 30% of the bounding box
                            face_height = int((y2 - y1) * 0.3)
                            face_y1 = y1
                            face_y2 = y1 + face_height
                            
                            # Ensure coordinates are within frame bounds
                            if face_y2 <= frame.shape[0] and x2 <= frame.shape[1]:
                                face_img = frame[face_y1:face_y2, x1:x2]
                                
                                # Add face to recognition queue if it's large enough
                                if face_img.shape[0] > 20 and face_img.shape[1] > 20:
                                    # Generate hash for deduplication
                                    import hashlib
                                    face_hash = hashlib.md5(face_img.tobytes()).hexdigest()
                                    
                                    face_data = {
                                        "face": face_img,
                                        "camera_name": camera_config.get("name", camera_id),
                                        "hash": face_hash
                                    }
                                    success = add_face_to_queue(face_data)
                                    if success:
                                        faces_sent += 1
                        
                        # Log only if faces were actually sent
                        if faces_sent > 0:
                            logger.debug(f"{camera_id}: Sent {faces_sent} face(s) to recognition queue")
                    else:
                        camera_stats[camera_id]["persons_detected"] = 0
            except Exception as e:
                logger.error(f"{camera_id}: Error in YOLO processing: {e}")
                import traceback
                traceback.print_exc()
            
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Error in camera thread {camera_id}: {e}")
        camera_stats[camera_id]["status"] = "error"
    finally:
        # Cleanup: release video capture when thread stops
        cap.release()
        logger.info(f"Camera thread {camera_id} stopped")
        camera_stats[camera_id]["status"] = "inactive"

def start_camera_threads():
    """
    Start processing threads for all configured cameras.
    Each camera runs in its own thread to allow parallel processing.
    """
    global camera_threads, stop_threads
    
    stop_threads = False
    cameras = CONFIG.get("cameras", [])
    
    if not cameras:
        logger.warning("No cameras configured")
        return
    
    # Filter only enabled cameras
    enabled_cameras = [c for c in cameras if c.get("enabled", False)]
    
    if not enabled_cameras:
        logger.warning("No cameras enabled in configuration")
        return
    
    logger.info(f"Starting {len(enabled_cameras)} enabled camera(s)")
    
    for camera in enabled_cameras:
        camera_id = camera.get("id")
        camera_name = camera.get("name", camera_id)
        
        if not camera_id:
            logger.warning("Camera configuration missing 'id', skipping")
            continue
        
        # Create and start thread for this camera
        thread = threading.Thread(
            target=process_camera_stream,
            args=(camera_id, camera),
            daemon=True,  # Daemon threads will exit when main program exits
            name=f"Camera-{camera_name}"
        )
        thread.start()
        camera_threads[camera_id] = thread
        
        logger.info(f"Started thread for camera {camera_id}")
    
    logger.info("All camera threads started")

def stop_camera_threads():
    """
    Stop all camera processing threads gracefully.
    """
    global stop_threads, camera_threads
    
    logger.info("Stopping camera threads...")
    stop_threads = True
    
    # Wait for all threads to finish (with timeout)
    for camera_id, thread in camera_threads.items():
        thread.join(timeout=5.0)
        if thread.is_alive():
            logger.warning(f"Camera thread {camera_id} did not stop gracefully")
    
    camera_threads.clear()
    logger.info("All camera threads stopped")

def get_camera_frame(camera_id):
    """
    Get the latest frame from a specific camera.
    
    Args:
        camera_id (str): Camera identifier
        
    Returns:
        numpy.ndarray: Latest frame from the camera, or None if not available
    """
    return camera_frames.get(camera_id)

def get_camera_stats(camera_id=None):
    """
    Get statistics for one or all cameras.
    
    Args:
        camera_id (str, optional): Specific camera ID, or None for all cameras
        
    Returns:
        dict: Camera statistics
    """
    if camera_id:
        return camera_stats.get(camera_id, {})
    return camera_stats.copy()
