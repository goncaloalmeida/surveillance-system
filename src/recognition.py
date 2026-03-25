"""
Facial Recognition Module
Handles face detection, embedding generation, and person identification
Uses DeepFace for embeddings and maintains a cooldown system to prevent spam detections
"""
import threading
import queue
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils import load_config
from src.logger import logger

CONFIG = load_config()

# Configuration constants
REFERENCE_FACES_DIR = Path('reference_faces')
MODEL = CONFIG.get("recognition").get("model")
MODELS = CONFIG.get("recognition").get("models", [])  # List of models for ensemble
THRESHOLD = CONFIG.get("recognition").get("threshold")
COOLDOWN_MINUTES = CONFIG.get("recognition").get("cooldown_minutes")
DETECTOR_BACKEND = CONFIG.get("recognition").get("detector_backend")
DISTANCE_METRIC = CONFIG.get("recognition").get("distance_metric")
CACHE_FILE = Path('temp/face_embeddings_cache.json')
FACE_EVENTS_FILE = Path('temp/face_events.json')
WORKERS = CONFIG.get("performance").get("workers")
EMBEDDING_WORKERS = CONFIG.get("performance").get("embedding_workers")
SAVE_UNRECOGNIZED = CONFIG.get("advanced").get("save_unrecognized_faces")
UNRECOGNIZED_THRESHOLD = CONFIG.get("advanced").get("unrecognized_threshold")

# Use ensemble mode only when more than one model is explicitly configured.
USE_ENSEMBLE = len(MODELS) > 1

# Global state
known_face_names = []  # List of recognized person names
face_embeddings_cache = {}  # name -> embedding vector mapping
identified_people_cache = {}  # camera_id -> {recognition_id -> person_data}
people_cooldown = {}  # person_name -> last_detection_timestamp
processed_face_hashes = {}  # hash -> timestamp (to avoid reprocessing same face)
cooldown_lock = threading.Lock()  # Thread safety for cooldown dict
deepface_lock = threading.Lock()  # Thread safety for DeepFace calls (TensorFlow not thread-safe)
recognition_queue = queue.Queue(maxsize=500)  # Queue for recognition tasks
worker_threads = []  # Worker thread references

def load_known_faces():
    """
    Load reference images from reference_faces/ and generate embeddings.
    Uses caching to avoid re-processing unchanged files.
    Processes multiple identities in parallel for speed.
    
    Returns:
        int: Number of faces successfully loaded
    """
    from deepface import DeepFace
    global face_embeddings_cache, known_face_names

    face_embeddings_cache = {}
    known_face_names = []
    
    logger.info(f"Loading reference images from {REFERENCE_FACES_DIR} using model {MODEL}")
    
    REFERENCE_FACES_DIR.mkdir(parents=True, exist_ok=True)
    face_files = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.webp'):
        face_files.extend(REFERENCE_FACES_DIR.glob(pattern))
    face_files = sorted(face_files)
    
    if not face_files:
        logger.warning("No reference images found in reference_faces/ directory")
        return 0
    
    logger.info(f"Found {len(face_files)} reference image(s)")
    logger.debug(f"Reference files: {[f.name for f in face_files]}")
    
    # Load existing cache
    cached_embeddings = {}
    logger.debug(f"Checking for cache file: {CACHE_FILE}")
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                cached_embeddings = cache_data.get('embeddings', {})
                logger.info(f"Loaded cache: {len(cached_embeddings)} embeddings, model={cache_data.get('model')}")
                logger.debug(f"Cached names: {list(cached_embeddings.keys())}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    else:
        logger.info("No cache file found - will process all photos")
    
    # Determine which files need processing (new or modified)
    photos_to_process = []
    for face_file in face_files:
        person_name = face_file.stem
        
        # Skip public template files.
        if person_name.startswith('EXAMPLE') or person_name.startswith('SAMPLE') or person_name.startswith('TEMPLATE'):
            continue
        
        file_mtime = os.path.getmtime(str(face_file))
        
        # Check if photo is new or modified
        if person_name not in cached_embeddings:
            photos_to_process.append(face_file)
        elif cached_embeddings[person_name].get('mtime') != file_mtime:
            photos_to_process.append(face_file)
    
    # Process new or modified files in parallel.
    if photos_to_process:
        logger.info(f"Processing {len(photos_to_process)} new or updated reference image(s)...")
        
        def process_embedding(face_file):
            """Generate an embedding for a single reference image."""
            try:
                person_name = face_file.stem
                
                # DeepFace.represent returns list of face embeddings found in image
                embedding = DeepFace.represent(
                    str(face_file), 
                    model_name=MODEL, 
                    detector_backend=DETECTOR_BACKEND,
                    enforce_detection=False
                )
                if embedding:
                    return (person_name, embedding[0]['embedding'], os.path.getmtime(str(face_file)))
                else:
                    logger.error(f"No embedding returned for {person_name}")
            except Exception as e:
                logger.error(f"Failed processing {face_file.name}: {type(e).__name__}: {e}", exc_info=True)
            return None
        
        # Use embedding_workers from config (default: 1 to prevent memory crashes)
        # Can be increased if system has sufficient RAM (each worker loads full TF model)
        
        with ThreadPoolExecutor(max_workers=EMBEDDING_WORKERS) as executor:
            futures = [executor.submit(process_embedding, f) for f in photos_to_process]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    name, embedding, mtime = result
                    # Store with model-specific key for cache compatibility
                    cached_embeddings[name] = {
                        f"{MODEL.lower().replace('-', '')}_embedding": embedding,
                        'mtime': mtime
                    }
        
        # Save updated cache
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, 'w') as f:
                json.dump({'version': '3.0', 'model': MODEL, 'embeddings': cached_embeddings}, f)
            logger.info("Cache updated successfully")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    # Load embeddings into memory for fast recognition.
    logger.info(f"Loading embeddings into memory...")
    for person_name, data in cached_embeddings.items():
        if USE_ENSEMBLE:
            # Keep full per-model data when running in ensemble mode.
            face_embeddings_cache[person_name] = data
            known_face_names.append(person_name)
            continue

        embedding_key = f"{MODEL.lower().replace('-', '')}_embedding"
        if embedding_key in data:
            face_embeddings_cache[person_name] = data[embedding_key]
            known_face_names.append(person_name)
        else:
            logger.warning(f"Skipped {person_name} - no embedding for {MODEL} (available: {list(data.keys())})")
    
    logger.info(f"Loaded {len(known_face_names)} reference identities with {MODEL}")
    logger.info(f"Known identities: {known_face_names}")
    return len(known_face_names)

def recognition_worker():
    """
    Worker thread that processes face recognition tasks from queue
    Compares detected faces against known embeddings using cosine distance
    Implements cooldown system to prevent duplicate detections
    """
    thread_name = threading.current_thread().name
    logger.info(f"[{thread_name}] Recognition worker started")
    
    try:
        from deepface import DeepFace
        from scipy.spatial import distance
    except Exception as import_error:
        logger.error(f"[{thread_name}] Failed to import: {import_error}", exc_info=True)
        return
    
    global processed_face_hashes
    
    while True:
        try:
            item = recognition_queue.get(timeout=1)
            
            if item is None:
                recognition_queue.task_done()
                break
            
            face_region, camera_name, face_hash = item['face'], item['camera_name'], item['hash']
            
            logger.info(f"[{thread_name}] Processing face from {camera_name}")
            
            try:
                # Skip if this exact face was recently processed (within cooldown period)
                with cooldown_lock:
                    current_time = datetime.now().timestamp()
                    if face_hash in processed_face_hashes:
                        time_since_processed = current_time - processed_face_hashes[face_hash]
                        if time_since_processed < (COOLDOWN_MINUTES * 60):
                            logger.debug(f"Skipping already processed face (hash: {face_hash[:8]}...)")
                            continue
                    # Mark this hash as processed
                    processed_face_hashes[face_hash] = current_time
                    
                    # Clean old hashes (older than 2x cooldown) - use list comprehension to keep in same dict
                    old_threshold = current_time - (COOLDOWN_MINUTES * 60 * 2)
                    hashes_to_remove = [h for h, t in processed_face_hashes.items() if t <= old_threshold]
                    for h in hashes_to_remove:
                        del processed_face_hashes[h]
                
                # Generate embedding for detected face
                with deepface_lock:
                    if USE_ENSEMBLE:
                        # Use multiple models and voting system
                        embeddings = {}
                        for model in MODELS:
                            try:
                                emb = DeepFace.represent(
                                    face_region, 
                                    model_name=model, 
                                    enforce_detection=False,
                                    detector_backend=DETECTOR_BACKEND
                                )
                                if emb:
                                    embeddings[model] = emb[0]['embedding']
                            except Exception as e:
                                logger.debug(f"Model {model} failed: {e}")
                        
                        if not embeddings:
                            continue
                        
                        # Voting: each model votes for best match
                        votes = {}
                        for model, face_embedding in embeddings.items():
                            model_key = f"{model.lower().replace('-', '')}_embedding"
                            best_for_model = None
                            best_dist = float('inf')
                            
                            for known_name, known_data in face_embeddings_cache.items():
                                if model_key not in known_data:
                                    continue
                                known_vector = known_data[model_key]
                                dist = distance.cosine(face_embedding, known_vector)
                                if dist < THRESHOLD and dist < best_dist:
                                    best_for_model = known_name
                                    best_dist = dist
                            
                            if best_for_model:
                                votes[best_for_model] = votes.get(best_for_model, 0) + 1
                        
                        # Best match is person with most votes
                        if votes:
                            best_match = max(votes, key=votes.get)
                            best_distance = 1.0 - (votes[best_match] / len(MODELS))  # Confidence score
                        else:
                            best_match = None
                            best_distance = float('inf')
                    else:
                        # Single model mode (default)
                        logger.debug(f"Single model mode: {MODEL}, detector={DETECTOR_BACKEND}")
                        logger.debug(f"Calling DeepFace.represent()...")
                        try:
                            embedding = DeepFace.represent(
                                face_region, 
                                model_name=MODEL, 
                                enforce_detection=False,
                                detector_backend=DETECTOR_BACKEND
                            )
                            logger.debug(f"DeepFace.represent() returned: {type(embedding)}, len={len(embedding) if embedding else 0}")
                        except Exception as deepface_error:
                            logger.error(f"DeepFace.represent() failed: {deepface_error}", exc_info=True)
                            continue
                            
                        if not embedding:
                            logger.debug("No embedding generated from face region")
                            continue
                        
                        face_embedding = embedding[0]['embedding']
                        logger.debug(f"Generated face embedding: {len(face_embedding)} dimensions")
                        
                        # Find best match in known faces
                        best_match = None
                        best_distance = float('inf')
                        
                        logger.debug(f"Comparing against {len(face_embeddings_cache)} known faces")
                        logger.debug(f"Known faces in cache: {list(face_embeddings_cache.keys())}")
                        
                        for known_name, known_vector in face_embeddings_cache.items():
                            # Keep compatibility with older cache formats.
                            if isinstance(known_vector, dict):
                                known_vector = known_vector.get(f"{MODEL.lower().replace('-', '')}_embedding")
                                if known_vector is None:
                                    continue
                            logger.debug(f"  Comparing with {known_name}: vector len={len(known_vector)}")
                            dist = distance.cosine(face_embedding, known_vector)
                            logger.debug(f"    Distance to {known_name}: {dist:.4f} (threshold: {THRESHOLD})")
                            if dist < THRESHOLD and dist < best_distance:
                                best_match = known_name
                                best_distance = dist
                                logger.debug(f"New best match: {known_name} (dist={dist:.4f})")
                        
                        if best_match:
                            logger.debug(f"FINAL MATCH: {best_match} with distance {best_distance:.4f}")
                        else:
                            logger.debug(f"NO MATCH FOUND (best distance was {best_distance:.4f}, threshold is {THRESHOLD})")
                
                if best_match:
                    # Check cooldown to avoid spam detections
                    with cooldown_lock:
                        if is_person_in_cooldown(best_match, camera_name):
                            logger.debug(f"{best_match} still in cooldown period, skipping")
                            continue
                        
                        # Update cooldown timestamp BEFORE processing to prevent race conditions
                        people_cooldown[best_match] = datetime.now().timestamp()
                    
                    # Store recognition in cache
                    recognition_id = f"{best_match}_{int(datetime.now().timestamp())}"
                    if camera_name not in identified_people_cache:
                        identified_people_cache[camera_name] = {}
                    
                    identified_people_cache[camera_name][recognition_id] = {
                        'name': best_match,
                        'first_seen': datetime.now().timestamp(),
                        'face_hash': face_hash
                    }
                    
                    # Save event and log ONCE
                    save_face_recognition_event(best_match, camera_name)
                    confidence = (1 - best_distance) * 100
                    logger.info(f"RECOGNIZED: {best_match} at {camera_name} (confidence: {confidence:.1f}%)")
                else:
                    # Face detected but not recognized - log for debugging
                    closest_name = None
                    closest_dist = float('inf')
                    
                    if len(face_embeddings_cache) > 0:
                        for known_name, known_vector in face_embeddings_cache.items():
                            if isinstance(known_vector, dict):
                                known_vector = known_vector.get(f"{MODEL.lower().replace('-', '')}_embedding")
                                if known_vector is None:
                                    continue
                            dist = distance.cosine(face_embedding, known_vector)
                            if dist < closest_dist:
                                closest_name = known_name
                                closest_dist = dist
                        
                        if closest_name:
                            confidence = (1 - closest_dist) * 100
                            logger.debug(f"Unrecognized face - closest match: {closest_name} (distance: {closest_dist:.3f}, {confidence:.1f}%)")
                    
            except Exception as e:
                logger.error(f"[{thread_name}] Error while processing face: {e}", exc_info=True)
            finally:
                recognition_queue.task_done()

                
        except queue.Empty:
            # Normal timeout when queue is empty.
            logger.debug(f"[{thread_name}] Queue timeout (normal)")
            continue
        except Exception as e:
            logger.error(f"[{thread_name}] Error in main loop: {e}", exc_info=True)

def start_recognition_workers():
    """
    Start background worker threads for face recognition
    Number of workers determined by config (default: 4)
    """
    global worker_threads
    logger.info(f"Starting {WORKERS} recognition workers...")
    
    for i in range(WORKERS):
        thread = threading.Thread(target=recognition_worker, daemon=True, name=f"RecognitionWorker-{i+1}")
        thread.start()
        worker_threads.append(thread)
    
    # Small pause to ensure worker threads are fully started.
    time.sleep(0.5)
    logger.info(f"Started {WORKERS} workers (threads alive: {sum(1 for t in worker_threads if t.is_alive())})")
    logger.info(f"Started {WORKERS} recognition workers")

def is_person_in_cooldown(person_name, camera_name):
    """
    Check if person is in cooldown period to prevent duplicate detections
    
    Args:
        person_name (str): Name of detected person
        camera_name (str): Camera that detected the person
    
    Returns:
        bool: True if in cooldown, False if can be detected again
    """
    if person_name not in people_cooldown:
        return False
    
    time_since_last = datetime.now().timestamp() - people_cooldown[person_name]
    return time_since_last < (COOLDOWN_MINUTES * 60)

def save_face_recognition_event(person_name, camera_name):
    """
    Save face recognition event to JSON file for history/analytics
    Keeps only last 100 events to prevent file from growing indefinitely
    
    Args:
        person_name (str): Name of recognized person
        camera_name (str): Camera where person was detected
    """
    try:
        events = []
        if FACE_EVENTS_FILE.exists():
            with open(FACE_EVENTS_FILE, 'r') as f:
                events = json.load(f)
        
        # Add new event at beginning of list
        events.insert(0, {
            'person': person_name,
            'camera': camera_name,
            'timestamp': datetime.now().isoformat(),
            'event_type': 'entry'  # Changed from 'type' to 'event_type' to match API expectations
        })
        
        # Keep only last 100 events
        FACE_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FACE_EVENTS_FILE, 'w') as f:
            json.dump(events[:100], f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to save event: {e}")

def get_face_events():
    """
    Retrieve face recognition events from storage
    
    Returns:
        list: List of event dictionaries, empty list if no events
    """
    try:
        if FACE_EVENTS_FILE.exists():
            with open(FACE_EVENTS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load events: {e}")
    return []
