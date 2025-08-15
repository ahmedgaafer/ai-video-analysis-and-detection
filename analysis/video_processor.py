import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Callable
import logging

from config import AIDetectionConfig

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video processing operations"""
    
    def __init__(self, config: AIDetectionConfig):
        self.config = config
        
    def sample_frames(self, cap: cv2.VideoCapture, progress_callback: Optional[Callable[[float, str], None]] = None) -> List[np.ndarray]:
        """Sample frames from video for processing"""
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate actual sample rate to stay within max_frames
        actual_sample_rate = max(1, total_frames // self.config.max_frames)
        sample_rate = max(self.config.sample_rate, actual_sample_rate)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while len(frames) < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                frames.append(frame)
                if progress_callback:
                    # Up to 30% of overall progress while sampling frames
                    progress = 5.0 + 25.0 * (len(frames) / max(1, min(self.config.max_frames, total_frames // sample_rate)))
                    progress_callback(min(progress, 30.0), "Sampling frames")
                
            frame_count += 1
            
        logger.info(f"Sampled {len(frames)} frames from {total_frames} total frames")
        return frames
    

    def get_video_capture(self, video_path: str) -> cv2.VideoCapture:
        """Get video capture object"""
        return cv2.VideoCapture(video_path)

    def validate_video(self, video_path: str) -> bool:
        """Validate video file"""
        if not Path(video_path).exists():
            logger.error(f"Video file does not exist: {video_path}")
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return False
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < self.config.min_frames:
            logger.error(f"Video has insufficient frames: {frame_count} < {self.config.min_frames}")
            cap.release()
            return False
            
        cap.release()
        return True 