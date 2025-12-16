import sys
import os
import json
import torch
from typing import Optional, Dict, Any, List, Union

class OutputDirs:
    """Output directories structure for experiments"""
    def __init__(self, root_dir: str, timestamp: str):
        """Initializes the output directory structure.
        
        Args:
            root_dir: Root directory for all outputs
            timestamp: Timestamp for the experiment
        """
        self.ROOT = os.path.join(root_dir, 'output', timestamp)
        os.makedirs(self.ROOT, exist_ok=True)
        
        self.LOGS = os.path.join(self.ROOT, 'logs')
        os.makedirs(self.LOGS, exist_ok=True)
        
        self.MODEL = os.path.join(self.ROOT, 'model')
        os.makedirs(self.MODEL, exist_ok=True)
        
        self.CKPT = os.path.join(self.ROOT, 'ckpt')
        os.makedirs(self.CKPT, exist_ok=True)
        
        self.SAMPLES = os.path.join(self.ROOT, 'samples')
        os.makedirs(self.SAMPLES, exist_ok=True)

class Experiment:
    """Class representing an experiment configuration"""
    def __init__(self, experiment_root: str, experiment_path: str, timestamp: str):
        """Initialize experiment from configuration file.
        
        Args:
            experiment_root: Root directory for the experiment
            experiment_path: Path to the experiment configuration
            timestamp: Timestamp for this run
        """
        self.PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
        self.EXPERIMENT_ROOT = experiment_root
        self.STRINGNAME = experiment_path.replace('/', '_')
        self.MODEL_NAME = experiment_path.split('/')[0]
        self.PARSEMODE = 'alt'  # Default parse mode
        self.TIMESTAMP = timestamp
        self.ENVIRONMENT = None
        
        # Setup output directories
        self.output = OutputDirs(experiment_root, timestamp)
        
        # Read experiment configuration
        config_file = os.path.join(experiment_root, 'experiment.json')
        assert os.path.exists(config_file), f"Experiment config file not found: {config_file}"
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract configuration values with defaults
        self.TILESIZE = config.get('tilesize', 256)
        self.IMG_HEIGHT = config.get('img_height', self.TILESIZE)
        self.IMG_WIDTH = config.get('img_width', self.TILESIZE)
        self.INPUT_CHANNELS = config.get('input_channels', 13)
        self.OUTPUT_CHANNELS = config.get('output_channels', 3)
        self.DATA_ROOT = config.get('data_root', None)
        self.ENVIRONMENT = config.get('environment', None)
        self.BATCH_SIZE = config.get('batch_size', 16)
        self.GEN_LOSS = config.get('gen_loss', {'gen_l1': 1})
        self.DISC_LOSS = config.get('disc_loss', {'disc_bce': 1})
        self.STEPS = config.get('steps', 40000)
        self.SHUFFLE = config.get('shuffle', True)
        self.MAX_SHUFFLE_BUFFER = config.get('max_shuffle_buffer', 500)
        self.RANDOM_ROTATE = config.get('random_rotate', False)
        self.RANDOM_RESIZE = config.get('random_resize', 1.1)
        self.PARSEMODE = config.get('parsemode', 'alt')
        self.DATA_SAMPLE = config.get('data_sample', None)
        self.EXCLUDE_SUFFIX = config.get('exclude_suffix', None)
        self.ENFORCE_SUFFIX = config.get('enforce_suffix', None)