"""
Utility Functions
Configuration loading and environment setup
"""
import os
import sys
import json
from pathlib import Path

# Configure UTF-8 encoding for Windows.
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

def load_config():
    """
    Load system configuration from config.json
    
    Returns:
        dict: Configuration dictionary with all system settings
    
    Raises:
        FileNotFoundError: If config.json is missing
        JSONDecodeError: If config.json has invalid JSON
    """
    config_path = Path(__file__).parent.parent / "config.json"
    
    if not config_path.exists():
        sys.stderr.write(f"ERROR: config.json not found at {config_path}\n")
        sys.stderr.write("Please create config.json with all required settings\n")
        sys.exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # Filter out documentation keys (starting with _)
            return {k: v for k, v in config.items() if not k.startswith('_')}
    except json.JSONDecodeError as e:
        sys.stderr.write(f"ERROR: Invalid JSON in config.json: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"ERROR: Failed to load config.json: {e}\n")
        sys.exit(1)
