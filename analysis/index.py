import logging
from typing import Dict, Optional, Callable
import pandas as pd

from config import AIDetectionConfig
from video_processor import VideoProcessor
from metrics import AIDetectionMetrics

import os
import concurrent.futures

import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_and_report(video_path: str, config: AIDetectionConfig, progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict:
    """
    Complete analysis with detailed reporting.
    
    Args:
        video_path: Path to the video file to analyze
        config: Optional configuration for analysis parameters
        
    Returns:
        Complete analysis report dictionary
    """

    video_processor = VideoProcessor(config)

    video_capture = video_processor.get_video_capture(video_path)
    if progress_callback:
        progress_callback(0.0, "Opening video")
    video_capture_frames = video_processor.sample_frames(video_capture, progress_callback=progress_callback)

    metrics = AIDetectionMetrics()

    # Support both new and old analyze() signatures
    try:
        return metrics.analyze(video_capture_frames, video_capture, progress_callback=progress_callback)
    except TypeError:
        # Fallback: run metrics step-by-step here to provide real progress updates
        if progress_callback:
            progress_callback(30.0, "Starting metrics")

        results: Dict = {}
        steps = [
            ("Brightness consistency", metrics.brightness_consistency_score, "brightness_consistency_score"),
            ("Temporal consistency", metrics.temporal_consistency_score, "temporal_consistency_score"),
            ("Motion field", metrics.motion_field_score, "motion_field_score"),
            ("Boundary stability", metrics.boundary_stability_score, "boundary_stability_score"),
            ("Color consistency", metrics.color_consistency_score, "color_consistency_score"),
            ("Scene smoothness", metrics.scene_smoothness_score, "scene_smoothness_score"),
            ("Background stability", metrics.background_stability_score, "background_stability_score"),
            ("Frequency anomalies", metrics.frequency_anomaly_score, "frequency_anomaly_score"),
            ("Noise patterns", metrics.noise_pattern_score, "noise_pattern_score"),
            ("Gradient consistency", metrics.gradient_consistency_score, "gradient_consistency_score"),
            ("Lighting consistency", metrics.lighting_consistency_score, "lighting_consistency_score"),
            ("Texture authenticity", metrics.texture_authenticity_score, "texture_authenticity_score"),
            ("Shadow consistency", metrics.shadow_consistency_score, "shadow_consistency_score"),
            ("Optical flow distribution", metrics.optical_flow_distribution_score, "optical_flow_distribution_score"),
            ("Perceptual hash", metrics.perceptual_hash_score, "perceptual_hash_score"),
        ]

        total = float(len(steps))
        for idx, (label, fn, key) in enumerate(steps, start=1):
            try:
                if progress_callback:
                    pct = 30.0 + (idx - 1) / total * 65.0
                    progress_callback(pct, label)
                value = fn(video_capture_frames)
                results[key] = value
            except Exception as e:
                logger.warning(f"Metric '{label}' failed: {e}")
                # Provide safe default consistent with metrics.analyze fallback
                default = 0.0 if key.endswith("_score") and key in [
                    "motion_field_score",
                    "frequency_anomaly_score",
                    "noise_pattern_score",
                    "lighting_consistency_score",
                ] else 100.0
                results[key] = default

        # Attempt to clear caches if present
        try:
            metrics.motion_cache.clear()
            metrics.reference_frames.clear()
        except Exception:
            pass

        if progress_callback:
            progress_callback(95.0, "Finalizing")

        return results

def process_video(path, config, is_ai):
    report = analyze_and_report(path, config)
    return report, is_ai, path

# Example usage
if __name__ == "__main__":
    
    real_video_base_path = "E:/master-dataset/real/"
    ai_video_base_path = "E:/master-dataset/ai/"
    
    real_video_paths = [os.path.join(real_video_base_path, f) for f in os.listdir(real_video_base_path)]
    ai_video_paths = [os.path.join(ai_video_base_path, f) for f in os.listdir(ai_video_base_path)]
    

    config = AIDetectionConfig(
            sample_rate=1,
            min_frames=3,
            max_frames=10000,
            chunk_size=100
        )

  

    start_time = time.time()
    print(f"Starting analysis at {start_time}")
    try:
        video_count = 6000
        tasks = []
        for i in range(video_count):
            real_video_path = real_video_paths[i]
            ai_video_path = ai_video_paths[i]
            tasks.append((real_video_path, config, False))
            tasks.append((ai_video_path, config, True))

        # Use max_workers for better resource control
        max_workers = min(len(tasks), int(os.cpu_count() or 1) -4)
        print(f"Using {max_workers} workers: {len(tasks)} : {os.cpu_count()}")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(process_video, path, conf, is_ai): (path, is_ai)
                for path, conf, is_ai in tasks
            }
            
            all_reports = []
            
            for future in concurrent.futures.as_completed(future_to_task):
                path, expected_is_ai = future_to_task[future]
                try:
                    report, is_ai, returned_path = future.result()
                    # Add the ai flag directly to the report
                    report['ai'] = 1 if is_ai else 0
                    all_reports.append(report)
                    print(f"Analyzed video: {returned_path}")
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    # Continue processing other videos instead of failing completely

        # More efficient DataFrame creation
        if all_reports:
            df = pd.DataFrame(all_reports)
            df.to_csv('reports.csv', index=False)
            print(f"Saved {len(all_reports)} reports to reports.csv")
        else:
            print("No reports generated")

    except Exception as e:
        print(f"Error: {e}")


    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    