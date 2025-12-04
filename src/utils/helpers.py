"""
Utility functions for the flight anomaly detection project.
"""
import yaml
import os
from pathlib import Path


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def setup_directories(config):
    """Create all necessary directories based on config."""
    root = get_project_root()
    
    # Data directories
    ensure_dir(root / config['data']['raw_dir'])
    ensure_dir(root / config['data']['processed_dir'])
    ensure_dir(root / config['data']['features_dir'])
    
    # Output directories
    ensure_dir(root / config['output']['figures_dir'])
    ensure_dir(root / config['output']['models_dir'])
    ensure_dir(root / config['output']['results_dir'])
    
    print("✓ All directories created successfully")
