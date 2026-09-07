import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import traceback
import numpy as np
import pandas as pd
import db_manager
import uuid

try:
    import yfinance as yf
except ImportError:
    print("yfinance is not installed. Please install it using: pip install yfinance")
    yf = None

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
except ImportError:
    print("matplotlib is not installed. Please install it using: pip install matplotlib")
    FigureCanvasTkAgg = None

try:
    import timesfm
    try:
        from timesfm3 import TimesFM3Evaluator, ModelConfig
        EVALUATOR_MODE = "timesfm3"
    except ImportError:
        EVALUATOR_MODE = "standard"
    TIMESFM_AVAILABLE = True
except ImportError:
    try:
        # Fallback just in case only timesfm3 is installed
        from timesfm3 import TimesFM3Evaluator, ModelConfig
        EVALUATOR_MODE = "timesfm3"
        TIMESFM_AVAILABLE = True
    except ImportError:
        print("timesfm is not installed or configured. Running in UI-only/Mock mode.")
        TIMESFM_AVAILABLE = False


class TimesFMApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TimesFM 3.0 Stock Data Forecaster")
        self.root.geometry("1400x850")
        self.root.minsize(1000, 700)
        
        # Internal state variables
        self.historical_data = None
        self.forecast_data = None
        self.is_processing = False
        
        # Model Caching State to prevent OOM/Bottlenecks
        self.loaded_model = None
        self.current_model_config = {}
        
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        self.setup_ui()
        self.log_message("System initialized. Ready to fetch data.")
        if not TIMESFM_AVAILABLE:
            self.log_message("WARNING: 'timesfm' module not found. Model will run in Mock Mode.", "warning")

        #  Initialize the database
        try: 
            db_manager.init_db()
            self.log_message("Database initialized successfully.")
        except Exception as e:
            self.log_message(f"Database initialization failed: {str(e)}", "error")

    def setup_ui(self):

        # Creating a parent notebook frame to hold the left and right panels
        parent_notebook = ttk.Notebook(self.root)
        parent_notebook.pack(fill="both", expand=True)

        #Creating the tabs inside the parent notebook
        tab_inference = ttk.Frame(parent_notebook)
        tab_logs = ttk.Frame(parent_notebook)
        tab_settings = ttk.Frame(parent_notebook)

        parent_notebook.add(tab_inference, text="Inference & Visualization")
        parent_notebook.add(tab_logs, text="System Logs")
        parent_notebook.add(tab_settings, text="Model Settings")

        left_panel = ttk.Frame(tab_inference, width=350, padding=(10, 10, 10, 10))
        left_panel.pack(side="left", fill="y", expand=False)
        
        
        canvas = tk.Canvas(left_panel)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        self.settings_frame = ttk.Frame(canvas)
        
        self.settings_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.build_data_settings()
        self.build_model_settings()
        self.build_action_buttons()
        
        right_panel = ttk.Frame(tab_inference, padding=(10, 10, 10, 10))
        right_panel.pack(side="right", fill="both", expand=True)
        right_panel.rowconfigure(0, weight=3) 
        right_panel.rowconfigure(1, weight=1) 
        right_panel.columnconfigure(0, weight=1)
        
        self.plot_frame = ttk.LabelFrame(right_panel, text="Data & Forecast Visualization")
        self.plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        self.log_frame = ttk.LabelFrame(right_panel, text="System Logs & Status")
        self.log_frame.grid(row=1, column=0, sticky="nsew")
        
        self.log_text = tk.Text(self.log_frame, state='disabled', wrap='word', height=8, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.init_plot()

    def build_data_settings(self):
        data_frame = ttk.LabelFrame(self.settings_frame, text="1. Data Farming (yfinance)", padding=(10, 5))
        data_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        ttk.Label(data_frame, text="Ticker Symbol:").grid(row=0, column=0, sticky="w", pady=2)
        self.tkr_var = tk.StringVar(value="ASELS.IS")
        ttk.Entry(data_frame, textvariable=self.tkr_var, width=15).grid(row=0, column=1, sticky="w", pady=2)
        
        ttk.Label(data_frame, text="Interval:").grid(row=1, column=0, sticky="w", pady=2)
        self.interval_var = tk.StringVar(value="1d")
        ttk.Combobox(data_frame, textvariable=self.interval_var, values=["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"], width=13, state="readonly").grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(data_frame, text="Data Period:").grid(row=2, column=0, sticky="w", pady=2)
        self.period_var = tk.StringVar(value="2y")
        ttk.Combobox(data_frame, textvariable=self.period_var, values=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], width=13, state="readonly").grid(row=2, column=1, sticky="w", pady=2)
        
        ttk.Label(data_frame, text="Target Column:").grid(row=3, column=0, sticky="w", pady=2)
        self.target_col_var = tk.StringVar(value="Close")
        ttk.Combobox(data_frame, textvariable=self.target_col_var, values=["Open", "High", "Low", "Close", "Volume"], width=13, state="readonly").grid(row=3, column=1, sticky="w", pady=2)

    def build_model_settings(self):
        model_frame = ttk.LabelFrame(self.settings_frame, text="2. TimesFM 3.0 Configuration", padding=(10, 5))
        model_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        ttk.Label(model_frame, text="HuggingFace Repo ID:").grid(row=0, column=0, columnspan=2, sticky="w", pady=(2,0))
        self.repo_var = tk.StringVar(value="google/timesfm-3.0-pytorch")
        ttk.Entry(model_frame, textvariable=self.repo_var, width=32).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0,5))
        
        ttk.Label(model_frame, text="Context Length (Input):").grid(row=2, column=0, sticky="w", pady=2)
        self.context_len_var = tk.IntVar(value=512)
        ttk.Entry(model_frame, textvariable=self.context_len_var, width=10).grid(row=2, column=1, sticky="e", pady=2)
        
        ttk.Label(model_frame, text="Horizon Length (Output):").grid(row=3, column=0, sticky="w", pady=2)
        self.horizon_var = tk.IntVar(value=64)
        ttk.Entry(model_frame, textvariable=self.horizon_var, width=10).grid(row=3, column=1, sticky="e", pady=2)
        
        ttk.Label(model_frame, text="Compute Backend:").grid(row=4, column=0, sticky="w", pady=2)
        self.backend_var = tk.StringVar(value="cpu")
        ttk.Combobox(model_frame, textvariable=self.backend_var, values=["cpu", "gpu", "tpu"], width=8, state="readonly").grid(row=4, column=1, sticky="e", pady=2)
        
        ttk.Label(model_frame, text="Freq Indicator (0=High, 1=Min...):").grid(row=5, column=0, sticky="w", pady=2)
        self.freq_ind_var = tk.IntVar(value=0)
        ttk.Entry(model_frame, textvariable=self.freq_ind_var, width=10).grid(row=5, column=1, sticky="e", pady=2)

        self.mock_mode_var = tk.BooleanVar(value=not TIMESFM_AVAILABLE)
        cb = ttk.Checkbutton(model_frame, text="Force Mock Mode (No GPU/API)", variable=self.mock_mode_var)
        cb.grid(row=6, column=0, columnspan=2, sticky="w", pady=5)
        if not TIMESFM_AVAILABLE:
            cb.state(['disabled'])

    def build_action_buttons(self):
        btn_frame = ttk.Frame(self.settings_frame, padding=(5, 5))
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.fetch_btn = ttk.Button(btn_frame, text="1. Fetch Data", command=self.thread_fetch_data)
        self.fetch_btn.pack(fill=tk.X, pady=2)
        
        self.run_btn = ttk.Button(btn_frame, text="2. Run TimesFM Forecast", command=self.thread_run_forecast)
        self.run_btn.pack(fill=tk.X, pady=2)
        
        self.save_btn = ttk.Button(btn_frame, text="3. Export Results as CSV", command=self.export_csv)
        self.save_btn.pack(fill=tk.X, pady=2)

    def init_plot(self):
        if FigureCanvasTkAgg is None:
            tk.Label(self.plot_frame, text="Matplotlib is required for plotting.").pack()
            return
            
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Historical Data & Forecast")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def log_message(self, message, level="info"):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, formatted_msg)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update_idletasks()

    def set_processing_state(self, state):
        self.is_processing = state
        if state:
            self.fetch_btn.state(['disabled'])
            self.run_btn.state(['disabled'])
        else:
            self.fetch_btn.state(['!disabled'])
            self.run_btn.state(['!disabled'])

    def thread_fetch_data(self):
        if self.is_processing: return
        self.set_processing_state(True)
        self.log_message(f"Fetching data for {self.tkr_var.get()}...")
        threading.Thread(target=self._fetch_data_job, daemon=True).start()

    def _fetch_data_job(self):
        try:
            if yf is None:
                raise ImportError("yfinance library is missing.")
                
            ticker = self.tkr_var.get().strip().upper()
            period = self.period_var.get()
            interval = self.interval_var.get()
            
            # Added timeout=10 to prevent network hanging that locks the UI thread permanently
            data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=10)
            
            if data.empty:
                raise ValueError(f"No data returned for ticker {ticker}.")
                
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                
            self.historical_data = data
            self.root.after(0, self._on_fetch_success)
        except Exception as e:
            self.root.after(0, self._on_process_error, str(e))

    def _on_fetch_success(self):
        self.log_message(f"Successfully fetched {len(self.historical_data)} data points.")
        self.update_plot()
        self.set_processing_state(False)

    def thread_run_forecast(self):
        if self.is_processing: return
        if self.historical_data is None or self.historical_data.empty:
            messagebox.showwarning("Warning", "Please fetch data first.")
            return
            
        self.set_processing_state(True)
        self.log_message("Initializing forecast sequence...")
        threading.Thread(target=self._run_forecast_job, daemon=True).start()

    def _run_forecast_job(self):
        try:
            target_col = self.target_col_var.get()
            if target_col not in self.historical_data.columns:
                raise ValueError(f"Column '{target_col}' not found in fetched data.")
                
            time_series = self.historical_data[target_col].values
            
            if np.isnan(time_series).all():
                raise ValueError(f"The column '{target_col}' contains only invalid/NaN data.")
            
            time_series = np.nan_to_num(time_series, nan=np.nanmean(time_series))
            
            context_len = self.context_len_var.get()
            horizon = self.horizon_var.get()
            
            if context_len <= 0:
                raise ValueError("Context length must be strictly greater than 0.")
                
            # FORCE CONTEXT LENGTH TO BE A MULTIPLE OF INPUT_PATCH_LEN (32) TO PREVENT TENSOR SHAPE MISMATCH
            input_patch_len = 32
            if context_len % input_patch_len != 0:
                old_len = context_len
                context_len = ((context_len // input_patch_len) + 1) * input_patch_len
                self.root.after(0, self.log_message, f"Adjusted Context Length from {old_len} to {context_len} (must be multiple of {input_patch_len}).")
            
            if len(time_series) < context_len:
                self.root.after(0, self.log_message, f"Warning: Data length ({len(time_series)}) < context length ({context_len}). Padding data.", "warning")
                pad_size = context_len - len(time_series)
                time_series = np.pad(time_series, (pad_size, 0), mode='edge')

            input_context = time_series[-context_len:]
            forecast_input = [input_context]
            
            is_mock = self.mock_mode_var.get()

            if is_mock or not TIMESFM_AVAILABLE:
                self.root.after(0, self.log_message, "Running in MOCK MODE (simulating model inference).")
                import time
                time.sleep(2)
                
                last_val = input_context[-1]
                volatility = np.std(input_context) * 0.1
                drift = (input_context[-1] - input_context[0]) / max(1, context_len)
                
                mock_forecast = [last_val + drift * i + np.random.normal(0, volatility) for i in range(1, horizon + 1)]
                forecast_result = np.array(mock_forecast)
                
            else:
                # IMPLEMENT MODEL CACHING TO PREVENT OOM CRASHES AND HEAVY BOTTLENECKS
                current_request_config = {
                    "repo": self.repo_var.get(),
                    "context_len": context_len,
                    "horizon": horizon,
                    "backend": self.backend_var.get()
                }
                
                if self.loaded_model is None or self.current_model_config != current_request_config:
                    self.root.after(0, self.log_message, f"Hardware loading TimesFM Model from {current_request_config['repo']} (this takes time)...")
                    
                    device_target = current_request_config["backend"]
                    import sys
                    if sys.platform == "darwin" and device_target == "gpu":
                        device_target = "mps"
                        self.root.after(0, self.log_message, "macOS detected: Mapped GPU to Apple MPS.")
                    
                    if EVALUATOR_MODE == "timesfm3":
                        # Actual TimesFM 3.0 Configuration
                        config = ModelConfig(
                            checkpoint_path=current_request_config["repo"],
                            device=device_target,
                            per_core_batch_size=1
                        )
                        self.loaded_model = TimesFM3Evaluator(config)
                    else:
                        self.loaded_model = timesfm.TimesFm(
                            context_len=context_len,
                            horizon_len=horizon,
                            input_patch_len=input_patch_len,
                            output_patch_len=128,
                            num_layers=20,
                            model_dims=1280,
                            backend=device_target
                        )
                        self.loaded_model.load_from_checkpoint(repo_id=current_request_config["repo"])
                        
                    self.current_model_config = current_request_config
                else:
                    self.root.after(0, self.log_message, "Using cached model weights in VRAM/RAM...")
                
                self.root.after(0, self.log_message, "Running inference on prepared context data...")
                
                # Run the actual prediction
                input_context = input_context.astype(np.float32)
                
                if EVALUATOR_MODE == "timesfm3":
                    # TimesFM 3.0 uses predict_batch instead of forecast
                    outputs = list(self.loaded_model.predict_batch(
                        [input_context],
                        horizon=horizon,
                        return_quantiles=False,
                        use_symmetric_averaging=False
                    ))
                    forecast_result = outputs[0].forecast
                else:
                    freq_ind = [self.freq_ind_var.get()]
                    point_forecast, _ = self.loaded_model.forecast([input_context], freq=freq_ind)
                    forecast_result = point_forecast[0]

            self.forecast_data = forecast_result


            # Save to database
            db_manager.insert_forecast(
                ticker=self.tkr_var.get(),
                interval=self.interval_var.get(),
                context_length=context_len,
                horizon_length=horizon,
                model_repo=self.repo_var.get(),
                period=self.period_var.get(),
                forecast_data=forecast_result.tolist(),
                mae_score=None  # Placeholder for MAE score, can be computed later if needed
            )

            self.root.after(0, self._on_forecast_success)
            
        except Exception as e:
            err_trace = traceback.format_exc()
            self.root.after(0, self._on_process_error, f"{str(e)}\n{err_trace}")

    def _on_forecast_success(self):
        self.log_message("Forecast completed successfully.")
        self.update_plot()
        self.set_processing_state(False)

    def _on_process_error(self, error_msg):
        self.log_message(f"ERROR: {error_msg}")
        messagebox.showerror("Process Error", str(error_msg))
        self.set_processing_state(False)

    def update_plot(self):
        if not hasattr(self, 'ax'): return
        
        self.ax.clear()
        self.ax.set_title(f"{self.tkr_var.get()} Historical & Forecast ({self.target_col_var.get()})")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        if self.historical_data is not None:
            target = self.target_col_var.get()
            dates = self.historical_data.index
            prices = self.historical_data[target]
            
            self.ax.plot(dates, prices, label="Historical Data", color="blue", linewidth=1.5)
            
            if self.forecast_data is not None:
                last_date = dates[-1]
                interval = self.interval_var.get()
                
                # FIXED LOGIC FLAW: Account for weekends on daily datasets
                if interval == "1d":
                    # Generate business days only (skips Saturday/Sunday)
                    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(self.forecast_data))
                else:
                    interval_map = {
                        "1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
                        "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
                        "1h": pd.Timedelta(hours=1),
                        "1wk": pd.Timedelta(weeks=1), "1mo": pd.DateOffset(months=1)
                    }
                    step = interval_map.get(interval, pd.Timedelta(days=1))
                    future_dates = [last_date + (step * i) for i in range(1, len(self.forecast_data) + 1)]
                
                connected_dates = [last_date] + list(future_dates)
                connected_prices = [prices.iloc[-1]] + list(self.forecast_data)
                
                self.ax.plot(connected_dates, connected_prices, label="TimesFM Forecast", color="orange", linewidth=2, linestyle="--")
                
            self.ax.legend()
            self.fig.autofmt_xdate()
            self.canvas.draw()

    def export_csv(self):
        if self.historical_data is None or self.forecast_data is None:
            messagebox.showinfo("Export", "No complete forecast data to export. Run a forecast first.")
            return
            
        import csv
        from tkinter import filedialog
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Forecast Data"
        )
        
        if not filepath:
            return
            
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Type", "Step", "Forecast_Value"])
                for i, val in enumerate(self.forecast_data):
                    writer.writerow(["Forecast", i+1, val])
            
            self.log_message(f"Forecast successfully exported to {filepath}")
        except Exception as e:
            self.log_message(f"Export Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TimesFMApp(root)
    root.mainloop()
