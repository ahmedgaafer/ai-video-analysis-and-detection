# AI Video Detector GUI

A graphical user interface for detecting AI-generated videos using advanced video analysis techniques.

## Features

- **Intuitive GUI**: Easy-to-use interface with file selection and analysis buttons
- **Radial Dial Display**: Visual representation of AI detection probability (0-100%)
- **Advanced Analysis**: Uses computer vision techniques including:
  - Temporal consistency analysis
  - Optical flow calculations
  - Compression artifact detection
  - Face consistency tracking
  - Object tracking validation
  - Texture authenticity assessment
- **Machine Learning**: XGBoost classifier for accurate predictions
- **Real-time Progress**: Progress bar and status updates during analysis

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the analysis modules in the `../analysis/` directory:
   - `index.py`
   - `config.py`
   - `video_processor.py`
   - `metrics.py`

## Usage

### Method 1: Direct execution
```bash
python index.py
```

### Method 2: Using the launcher
```bash
python run_gui.py
```

## How to Use

1. **Launch the Application**: Run the program using one of the methods above
2. **Select Video**: Click "Select Video" button to choose a video file
3. **Analyze**: Click "Analyze Video" to start the analysis process
4. **View Results**: The radial dial will show the AI detection probability:
   - **0-30%**: Likely Real Video (Green)
   - **30-70%**: Uncertain (Orange)
   - **70-100%**: Likely AI Generated (Red)

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)

## Technical Details

### Analysis Process
1. **Frame Sampling**: Extracts frames from the video at configurable intervals
2. **Feature Extraction**: Calculates multiple metrics including:
   - Temporal consistency
   - Motion patterns
   - Lighting consistency
   - Compression artifacts
   - Face detection and tracking
   - Object movement validation
3. **Machine Learning**: Uses XGBoost classifier for final prediction
4. **Visualization**: Displays results on a radial dial interface

### Model Training
- The application automatically loads a pre-trained model if available
- If training data exists (`reports.csv`), it will train a new model
- Falls back to heuristic-based detection if no model is available

## Configuration

The analysis parameters can be adjusted in the `AIDetectionConfig` class:
- `sample_rate`: Process every Nth frame (default: 1)
- `min_frames`: Minimum frames for analysis (default: 3)
- `max_frames`: Maximum frames to prevent memory issues (default: 100)
- `chunk_size`: Process frames in chunks (default: 50)

### GUI Options

- A dropdown allows choosing `sample_rate` (every Nth frame). Options: 1, 2, 3, 5, 10, 15, 20, 30, 50.
- A determinate progress bar shows progress up to 100% while analyzing.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed and the code directory structure is correct
2. **Memory Issues**: Reduce `max_frames` or `chunk_size` for large videos
3. **Slow Performance**: Increase `sample_rate` to process fewer frames
4. **File Not Found**: Ensure video file path is accessible and format is supported

### Performance Tips

- Use smaller video files for faster analysis
- Increase sample rate for quicker (but less accurate) results
- Close other applications to free up system resources

## License

This project is for educational and research purposes. 