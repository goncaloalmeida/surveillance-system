"""
Centralized Logging System
Provides unified logging with console/file/both output modes
"""
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

class AppLogger:
    """
    Singleton logger for the application
    Supports console, file, or both output modes
    Automatically cleans old log files based on max_age_days
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.logger = None
    
    def setup(self, config):
        """
        Configure logging based on config.json settings
        
        Args:
            config (dict): Configuration dictionary with logging settings
        """
        log_config = config.get('logging')
        mode = log_config.get('mode')
        level = log_config.get('level').upper()
        log_file = log_config.get('file')
        max_age_days = log_config.get('max_age_days')
        
        # Create logger
        self.logger = logging.getLogger('SurveillanceSystem')
        self.logger.setLevel(getattr(logging, level, logging.INFO))
        self.logger.handlers.clear()
        # Prevent logs from propagating to root logger (which may print to console)
        self.logger.propagate = False
        
        # Format: [2024-12-14 15:30:45] INFO - Message
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if mode in ['console', 'both']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler with rotation (10MB max per file, keep 5 backups)
        if mode in ['file', 'both']:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Clean old log files
            self._cleanup_old_logs(log_path.parent, max_age_days)
        
        self.info(f"Logging initialized: mode={mode}, level={level}")
    
    def _cleanup_old_logs(self, log_dir, max_age_days):
        """
        Remove log files older than max_age_days
        
        Args:
            log_dir (Path): Directory containing log files
            max_age_days (int): Maximum age of log files in days
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            for log_file in log_dir.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    self.debug(f"Removed old log file: {log_file.name}")
        except Exception as e:
            # Use print instead of logger to avoid recursion
            print(f"Warning: Failed to cleanup old logs: {e}")
    
    def debug(self, message):
        """Log debug message"""
        if self.logger:
            self.logger.debug(message)
    
    def info(self, message):
        """Log info message"""
        if self.logger:
            self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        if self.logger:
            self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        if self.logger:
            self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        if self.logger:
            self.logger.critical(message)

# Global logger instance
logger = AppLogger()
