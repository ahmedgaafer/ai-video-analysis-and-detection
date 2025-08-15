import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sys
import os
import math
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Add the parent directory to the path to import from code folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))

try:
    from index import analyze_and_report  # type: ignore
    from config import AIDetectionConfig  # type: ignore
except ImportError:
    # Fallback import method
    import importlib.util
    
    code_dir = os.path.join(os.path.dirname(__file__), '..', 'analysis')
    
    # Import analyze module
    analyze_spec = importlib.util.spec_from_file_location("analyze", os.path.join(code_dir, "index.py"))
    if analyze_spec and analyze_spec.loader:
        analyze_module = importlib.util.module_from_spec(analyze_spec)
        analyze_spec.loader.exec_module(analyze_module)
        analyze_and_report = analyze_module.analyze_and_report
    else:
        raise ImportError("Could not load analyze module")
    
    # Import config module
    config_spec = importlib.util.spec_from_file_location("config", os.path.join(code_dir, "config.py"))
    if config_spec and config_spec.loader:
        config_module = importlib.util.module_from_spec(config_spec)
        config_spec.loader.exec_module(config_module)
        AIDetectionConfig = config_module.AIDetectionConfig
    else:
        raise ImportError("Could not load config module")
try:
    import xgboost as xgb
except Exception:
    xgb = None
from sklearn.model_selection import train_test_split

class RadialDial:
    def __init__(self, canvas, x, y, radius, value=0):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.radius = radius
        self.value = value
        self.dial_id = None
        self.needle_id = None
        self.text_id = None
        self.create_dial()
    
    def create_dial(self):
        # Create the dial background
        self.canvas.create_oval(
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius,
            fill="lightgray", outline="black", width=2
        )
        
        # Create tick marks and labels
        for i in range(11):  # 0 to 100 in steps of 10
            # Correct sweep from 180° (left) to 0° (right)
            angle = math.pi - (i * math.pi / 10)
            tick_start_x = self.x + (self.radius - 20) * math.cos(angle)
            tick_start_y = self.y + (self.radius - 20) * -math.sin(angle)
            tick_end_x = self.x + (self.radius - 10) * math.cos(angle)
            tick_end_y = self.y + (self.radius - 10) * -math.sin(angle)
            
            self.canvas.create_line(
                tick_start_x, tick_start_y, tick_end_x, tick_end_y,
                fill="black", width=2
            )
            
            # Add labels
            label_x = self.x + (self.radius - 35) * math.cos(angle)
            label_y = self.y + (self.radius - 35) * -math.sin(angle)
            self.canvas.create_text(
                label_x, label_y, text=str(i * 10),
                font=("Arial", 10, "bold")
            )
        
        # Add AI/Real labels
        self.canvas.create_text(
            self.x - self.radius + 40, self.y + 20,
            text="REAL", font=("Arial", 12, "bold"), fill="green"
        )
        self.canvas.create_text(
            self.x + self.radius - 40, self.y + 20,
            text="AI", font=("Arial", 12, "bold"), fill="red"
        )
        
        # Create needle
        self.update_needle()
        
        # Create center circle
        self.canvas.create_oval(
            self.x - 8, self.y - 8, self.x + 8, self.y + 8,
            fill="black", outline="black"
        )
        
        # Create value text
        self.text_id = self.canvas.create_text(
            self.x, self.y + 40, text=f"{self.value:.1f}%",
            font=("Arial", 16, "bold"), fill="blue"
        )
    
    def update_needle(self):
        if self.needle_id:
            self.canvas.delete(self.needle_id)
        
        # Convert value (0-100) to angle (180° to 0°) with correct orientation
        angle = math.pi - (self.value / 100.0) * math.pi
        
        needle_x = self.x + (self.radius - 15) * math.cos(angle)
        needle_y = self.y + (self.radius - 15) * -math.sin(angle)
        
        self.needle_id = self.canvas.create_line(
            self.x, self.y, needle_x, needle_y,
            fill="red", width=4
        )
    
    def set_value(self, value):
        self.value = max(0, min(100, value))
        self.update_needle()
        if self.text_id:
            self.canvas.delete(self.text_id)
        self.text_id = self.canvas.create_text(
            self.x, self.y + 40, text=f"{self.value:.1f}%",
            font=("Arial", 16, "bold"), fill="blue"
        )

class AIVideoDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Video Detector")
        self.root.geometry("800x700")
        self.root.configure(bg="white")
        
        self.selected_video = None
        self.model = None
        self.model_trained = False
        self.config = AIDetectionConfig(
            sample_rate=1,
            min_frames=3,
            max_frames=100,
            chunk_size=50
        )
        
        self.create_widgets()
        self.load_or_train_model()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.root, text="AI Video Detector",
            font=("Arial", 24, "bold"), bg="white", fg="navy"
        )
        title_label.pack(pady=20)
        
        # File selection frame
        file_frame = tk.Frame(self.root, bg="white")
        file_frame.pack(pady=20)
        
        self.file_label = tk.Label(
            file_frame, text="No video selected",
            font=("Arial", 12), bg="white", fg="gray"
        )
        self.file_label.pack(side=tk.LEFT, padx=10)
        
        select_button = tk.Button(
            file_frame, text="Select Video",
            command=self.select_video,
            font=("Arial", 12), bg="lightblue", fg="black",
            relief=tk.RAISED, bd=2
        )
        select_button.pack(side=tk.LEFT, padx=10)
        
        # Analyze button
        self.analyze_button = tk.Button(
            self.root, text="Analyze Video",
            command=self.analyze_video,
            font=("Arial", 14, "bold"), bg="green", fg="white",
            relief=tk.RAISED, bd=3, state=tk.DISABLED
        )
        self.analyze_button.pack(pady=20)
        
        # Settings frame
        settings_frame = tk.Frame(self.root, bg="white")
        settings_frame.pack(pady=10)

        tk.Label(settings_frame, text="Sample rate (process every Nth frame):", bg="white", font=("Arial", 11)).pack(side=tk.LEFT, padx=(0, 8))
        self.sample_rate_var = tk.StringVar(value=str(self.config.sample_rate))
        self.sample_rate_combo = ttk.Combobox(settings_frame, textvariable=self.sample_rate_var, state="readonly", width=6,
                                              values=["1", "2", "3", "5", "10", "15", "20", "30", "50"])
        self.sample_rate_combo.pack(side=tk.LEFT)
        self.sample_rate_combo.bind("<<ComboboxSelected>>", self.on_sample_rate_change)

        # Progress bar (determinate)
        self.progress = ttk.Progressbar(
            self.root, length=400, mode='determinate', maximum=100
        )
        self.progress.pack(pady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.root, text="Ready", font=("Arial", 12),
            bg="white", fg="blue"
        )
        self.status_label.pack(pady=5)
        
        # Canvas for radial dial
        self.canvas = tk.Canvas(
            self.root, width=400, height=250,
            bg="white", highlightthickness=0
        )
        # Pack canvas first, then insert label before it
        self.canvas.pack(pady=20)

        # Result label (placed above the dial)
        self.result_label = tk.Label(
            self.root, text="", font=("Arial", 14, "bold"),
            bg="white"
        )
        self.result_label.pack(pady=20, before=self.canvas)

        # Create the dial
        self.dial = RadialDial(self.canvas, 200, 100, 100, 0)

        # Internal progress state
        self.is_analyzing = False
        self.progress_value = 0

    def on_sample_rate_change(self, event=None):
        try:
            new_rate = int(self.sample_rate_var.get())
            self.config.sample_rate = max(1, new_rate)
            self.status_label.config(text=f"Sample rate set to every {self.config.sample_rate} frame(s)")
        except ValueError:
            # Revert to previous valid value
            self.sample_rate_var.set(str(self.config.sample_rate))
    
    def select_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_video = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=filename, fg="black")
            self.analyze_button.config(state=tk.NORMAL)
            self.status_label.config(text="Video selected. Ready to analyze.")
    
    def load_or_train_model(self):
        """Load existing model or train new one if data is available"""
        model_path = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'ai_detector_model.pkl')
        reports_path = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'reports.csv')
        
        try:
            # Try to load existing model
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_trained = True
                self.status_label.config(text="Model loaded successfully")
                return
            
            # Try to train model if data exists
            if os.path.exists(reports_path) and xgb is not None:
                self.status_label.config(text="Training model...")
                self.root.update()
                
                df = pd.read_csv(reports_path)
                X = df.drop('ai', axis=1)
                y = df['ai']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                self.model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='auc',
                    use_label_encoder=False,
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1
                )
                
                self.model.fit(X_train, y_train)
                
                # Save model
                with open(model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                
                self.model_trained = True
                self.status_label.config(text="Model trained and saved successfully")
            elif os.path.exists(reports_path) and xgb is None:
                self.status_label.config(text="XGBoost not available; skipping training. Using heuristic.")
            else:
                self.status_label.config(text="No training data found. Model will be basic.")
                
        except Exception as e:
            self.status_label.config(text=f"Model loading error: {str(e)}")
    
    def analyze_video(self):
        if not self.selected_video:
            messagebox.showerror("Error", "Please select a video first")
            return
        
        self.analyze_button.config(state=tk.DISABLED)
        self.status_label.config(text="Analyzing video...")
        self.result_label.config(text="")
        # Start determinate progress animation up to 95%
        self.is_analyzing = True
        self.progress_value = 0
        self.progress['value'] = 0
        # Disable settings while analyzing
        self.sample_rate_combo.configure(state="disabled")
        self._tick_progress()
        
        # Run analysis in separate thread
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        try:
            # Analyze video
            def on_progress(pct, label):
                try:
                    # Clamp 0-100 and update from worker via main thread
                    pct = max(0.0, min(100.0, float(pct)))
                    self.root.after(0, self._set_progress_ui, pct, label)
                except Exception:
                    pass

            report = analyze_and_report(self.selected_video, self.config, progress_callback=on_progress)
            
            # Predict using model
            if self.model_trained and self.model:
                # Convert report to DataFrame
                report_df = pd.DataFrame([report])
                
                # Make prediction
                prediction_proba = self.model.predict_proba(report_df)[0]
                ai_probability = prediction_proba[1] * 100  # Convert to percentage
            else:
                # Basic heuristic if no model available
                ai_probability = self.basic_heuristic(report)
            
            # Update GUI in main thread
            self.root.after(0, self.update_results, ai_probability)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def _tick_progress(self):
        # Kept as a fallback in case no callbacks fire; slow creep up to 60%
        if self.is_analyzing:
            if self.progress_value < 60:
                self.progress_value += 0.5
                self.progress['value'] = self.progress_value
            self.root.after(200, self._tick_progress)

    def _set_progress_ui(self, pct: float, label: str):
        if self.is_analyzing:
            self.progress_value = pct
            self.progress['value'] = pct
            self.status_label.config(text=f"{label}... {int(pct)}%")
    
    def basic_heuristic(self, report):
        """Basic heuristic when no trained model is available"""
        # Simple heuristic based on some metrics
        # This is a placeholder - replace with your own logic
        score = 0
        
        # Example heuristic factors
        if 'temporal_consistency_mean' in report:
            if report['temporal_consistency_mean'] < 0.5:
                score += 30
        
        if 'compression_artifacts_mean' in report:
            if report['compression_artifacts_mean'] > 0.7:
                score += 25
        
        if 'optical_flow_magnitude_mean' in report:
            if report['optical_flow_magnitude_mean'] > 10:
                score += 20
        
        return min(score, 100)
    
    def update_results(self, ai_probability):
        # Finalize progress
        self.is_analyzing = False
        self.progress_value = 100
        self.progress['value'] = 100
        self.analyze_button.config(state=tk.NORMAL)
        self.sample_rate_combo.configure(state="readonly")
        
        # Update dial
        self.dial.set_value(ai_probability)
        
        # Update result label
        if ai_probability > 70:
            result_text = "LIKELY AI GENERATED"
            result_color = "red"
        elif ai_probability > 30:
            result_text = "UNCERTAIN"
            result_color = "orange"
        else:
            result_text = "LIKELY REAL"
            result_color = "green"
        
        self.result_label.config(text=result_text, fg=result_color)
        self.status_label.config(text="Analysis complete")
    
    def show_error(self, error_msg):
        self.is_analyzing = False
        self.analyze_button.config(state=tk.NORMAL)
        self.sample_rate_combo.configure(state="readonly")
        self.status_label.config(text="Error occurred")
        messagebox.showerror("Analysis Error", f"Error analyzing video:\n{error_msg}")

def main():
    root = tk.Tk()
    app = AIVideoDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
