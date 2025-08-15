## AI Video Analysis and Detection

A lightweight toolkit and GUI for analyzing videos and estimating the likelihood they are AI‑generated. It provides a simple desktop interface, configurable frame sampling, and a clear progress indicator while the analysis runs.

### What it does
- **Video analysis**: Extracts features from sampled frames and applies a model or heuristic to estimate AI‑generation probability.
- **GUI app**: Select a video, choose a sampling rate, track a real progress bar, and view the final likelihood on a radial dial.
- **Configurable**: Tune parameters via `analysis/config.py` and the GUI sampling‑rate dropdown.

### Quick start
1. Create a virtual environment (recommended) and install dependencies:
```bash
pip install -r detection/requirements.txt
```
2. Launch the GUI:
```bash
python detection/run_gui.py
```

### Using the GUI
- Click "Select Video" to choose a file.
- Adjust **Sample rate** (process every Nth frame). Higher values run faster but analyze fewer frames.
- Press **Analyze Video**. The status line shows the current stage (e.g., sampling frames, metric being computed), and the progress bar advances to 100%.
- The radial dial shows the estimated probability and a concise verdict label.

### Project layout
- `analysis/`: Core analysis modules, configuration, and utilities.
- `detection/`: End‑user GUI, launcher, and GUI‑specific docs.

### Minimal configuration
- Edit defaults in `analysis/config.py` (e.g., `sample_rate`, `max_frames`).
- Optionally provide `analysis/reports.csv` to train an XGBoost model automatically on first run; otherwise a safe heuristic is used.

### Troubleshooting
- If the model training step is unavailable (e.g., missing XGBoost), the app automatically falls back to a heuristic and continues.
- For large or long videos, increase `sample_rate` or lower `max_frames` to speed up analysis and reduce memory usage.

## How to Cite

If you use this software or build upon the ideas in your work, please cite the following manuscript:

**Ahmed Samy Gaafer**. **Unmasking Synthetic Realities: A Two-Fold Thesis on AI Video Analysis and Detection**.

### BibTeX
```bibtex
@unpublished{Gaafer2025Unmasking,
  author       = {Ahmed Samy Gaafer},
  title        = {Unmasking Synthetic Realities: A Two-Fold Thesis on AI Video Analysis and Detection},
  year         = {2025},
  note         = {Manuscript},
}
```

If you prefer to cite the software directly:
```bibtex
@software{Gaafer2025AIVideoAnalysisAndDetection,
  author = {Ahmed Samy Gaafer},
  title  = {AI Video Analysis and Detection},
  year   = {2025},
}
```


