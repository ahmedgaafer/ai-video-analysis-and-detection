from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class AIDetectionConfig:
    """Configuration class for AI detection parameters"""
    # Thresholds
    brightness_threshold: float = 50.0
    temporal_threshold: float = 50.0
    motion_threshold: float = 2.0
    boundary_threshold: float = 15.0
    color_threshold: float = 50.0
    fps_threshold: float = 20.0
    scene_threshold: float = 5.0
    background_threshold: float = 25.0
    frequency_threshold: float = 0.02
    noise_threshold: float = 50.0
    gradient_threshold: float = 20.0
    lighting_threshold: float = 1.5
    compression_threshold: float = 1000.0
    face_consistency_threshold: float = 75.0  
    object_tracking_threshold: float = 80.0   
    texture_authenticity_threshold: float = 60.0  
    shadow_consistency_threshold: float = 70.0     
    optical_flow_threshold: float = 0.15           
    blur_pattern_threshold: float = 0.8            
    perceptual_hash_threshold: float = 85.0        
    
    # Processing parameters
    sample_rate: int = 5  # Process every Nth frame
    min_frames: int = 2  # Minimum frames for analysis
    max_frames: int = 1000  # Maximum frames to prevent memory issues
    chunk_size: int = 100  # Process frames in chunks
    
    
    # Weights for final scoring
    weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {}


    