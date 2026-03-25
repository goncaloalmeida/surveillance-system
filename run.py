"""
Production Server Entry Point
Surveillance System with Facial Recognition
"""
from waitress import serve
from src.app import app, initialize
from src.logger import logger
from src.utils import load_config

CONFIG = load_config()

if __name__ == '__main__':
    # Initialize system (logging, cameras, recognition)
    faces_loaded = initialize()
    
    # Get server configuration
    server_config = CONFIG.get('server')
    host = server_config.get('host')
    port = server_config.get('port')
    threads = server_config.get('threads')
    
    logger.info(f"System ready: {faces_loaded} people loaded")
    logger.info(f"Starting server on http://{host}:{port}")
    logger.info("Press CTRL+C to stop")
    
    # Start production server with Waitress
    serve(app, host=host, port=port, threads=threads, channel_timeout=300)
