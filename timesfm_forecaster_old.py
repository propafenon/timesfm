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
    import matplotlib.dates as mdates
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
        print("timesfm is not installed or configured. Inference will fail.")
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
        # What the live forecast was conditioned on, captured at inference time
        # rather than at save time so it stays correct if the data is refetched.
        self.forecast_anchor = None
        self.forecast_target_col = None
        self.overlay_forecasts = []
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
            self.log_message("WARNING: 'timesfm' module not found. Inference will fail.", "warning")

        #  Initialize the database
        try: 
            db_manager.init_db()
            self.log_message("Database initialized successfully.")
        except Exception as e:
            self.log_message(f"Database initialization failed: {str(e)}", "error")

        self.refresh_forecast_history()

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

        # INFERENCE & VISUALIZATION TAB
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

        # LOGS AND QC TAB: tab_logs
        
        self.data_grid_frame = ttk.LabelFrame(tab_logs, text="Forecast History", padding=(10, 10, 10, 10))
        self.data_grid_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ## DATA GRID and children
        self.run_tree = ttk.Treeview(self.data_grid_frame, columns=("ID", "Timestamp", "Ticker", "Interval", "Context Length", "Horizon Length", "Model Repo", "Period", "Target", "Anchor Date", "MAE Score"), show="headings", selectmode="extended")
        self.run_tree.pack(fill="both", expand=True, side="left")
        self.run_tree.heading("ID", text="ID",)
        self.run_tree.heading("Timestamp", text="Timestamp")
        self.run_tree.heading("Ticker", text="Ticker")
        self.run_tree.heading("Interval", text="Interval")
        self.run_tree.heading("Context Length", text="Context Length")
        self.run_tree.heading("Horizon Length", text="Horizon Length")
        self.run_tree.heading("Model Repo", text="Model Repo")
        self.run_tree.heading("Period", text="Period")
        self.run_tree.heading("Target", text="Target")
        self.run_tree.heading("Anchor Date", text="Anchor Date")
        self.run_tree.heading("MAE Score", text="MAE Score")

        tree_scrollbar_horizontal = ttk.Scrollbar(self.data_grid_frame, orient="horizontal", command=self.run_tree.xview)
        tree_scrollbar_vertical = ttk.Scrollbar(self.data_grid_frame, orient="vertical", command=self.run_tree.yview)
        tree_scrollbar_horizontal.pack(side="bottom", fill="x")
        tree_scrollbar_vertical.pack(side="right", fill="y")
        self.run_tree.configure(xscrollcommand=tree_scrollbar_horizontal.set, yscrollcommand=tree_scrollbar_vertical.set)

        ### Configure treeview columns
        self.qc_controls_frame = ttk.LabelFrame(tab_logs, text="QC Controls")
        self.qc_controls_frame.pack(fill="x", pady=(0, 10), padx=10)
        


        
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
        self.period_var = tk.StringVar(value="max")
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
        self.context_len_var = tk.IntVar(value=1056)
        ttk.Entry(model_frame, textvariable=self.context_len_var, width=10).grid(row=2, column=1, sticky="e", pady=2)
        
        ttk.Label(model_frame, text="Horizon Length (Output):").grid(row=3, column=0, sticky="w", pady=2)
        self.horizon_var = tk.IntVar(value=7)
        ttk.Entry(model_frame, textvariable=self.horizon_var, width=10).grid(row=3, column=1, sticky="e", pady=2)
        
        ttk.Label(model_frame, text="Compute Backend:").grid(row=4, column=0, sticky="w", pady=2)
        self.backend_var = tk.StringVar(value="gpu")
        ttk.Combobox(model_frame, textvariable=self.backend_var, values=["cpu", "gpu", "tpu"], width=8, state="readonly").grid(row=4, column=1, sticky="e", pady=2)
        
        ttk.Label(model_frame, text="Freq Indicator (0=High, 1=Min...):").grid(row=5, column=0, sticky="w", pady=2)
        self.freq_ind_var = tk.IntVar(value=0)
        ttk.Entry(model_frame, textvariable=self.freq_ind_var, width=10).grid(row=5, column=1, sticky="e", pady=2)

    def build_action_buttons(self):

        ## SETTINGS_FRAME BUTTONS - TAB 1
        btn_frame = ttk.Frame(self.settings_frame, padding=(5, 5))
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.fetch_btn = ttk.Button(btn_frame, text="Fetch Data", command=self.thread_fetch_data)
        self.fetch_btn.pack(fill=tk.X, pady=2)
        
        self.run_btn = ttk.Button(btn_frame, text="Run Forecast", command=self.thread_run_forecast)
        self.run_btn.pack(fill=tk.X, pady=2)

        self.insert_db_btn = ttk.Button(btn_frame, text="Save Forecast to DB", command=self.save_forecast_to_db)
        self.insert_db_btn.pack(fill=tk.X, pady=2)
        
        self.save_btn = ttk.Button(btn_frame, text="Export Results as CSV", command=self.export_csv)
        self.save_btn.pack(fill=tk.X, pady=2)

        ## QC CONTROLS FRAME BUTTONS - TAB 2
        self.refresh_btn = ttk.Button(self.qc_controls_frame, text="Refresh", command=self.refresh_forecast_history, width=6)
        self.refresh_btn.pack(side=tk.LEFT, padx=5, pady=5, )

        self.overlay_btn = ttk.Button(self.qc_controls_frame, text="Overlay Selected Forecast", command=self.overlay_selected_forecast, width=18)
        self.overlay_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.calculate_mae_btn = ttk.Button(self.qc_controls_frame, text="Calculate MAE for Selected", command=self.calculate_mae_for_selected, width=20)
        self.calculate_mae_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.delete_btn = ttk.Button(self.qc_controls_frame, text="Delete Selected Forecast", command=self.delete_selected_forecast)
        self.delete_btn.pack(side=tk.LEFT, padx=5, pady=5)

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
        self.overlay_forecasts.clear()
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
            
            if not TIMESFM_AVAILABLE:
                raise ImportError("timesfm library is not available. Cannot run forecast.")
                
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
            self.forecast_anchor = self.historical_data.index[-1]
            self.forecast_target_col = target_col

            self.root.after(0, self._on_forecast_success)
            
        except Exception as e:
            err_trace = traceback.format_exc()
            self.root.after(0, self._on_process_error, f"{str(e)}\n{err_trace}")

    def _on_forecast_success(self):
        self.log_message("Forecast completed successfully.")
        self.update_plot()
        self.set_processing_state(False)

    def save_forecast_to_db(self):
        # 1. Safety check: make sure a forecast actually exists first!
        if self.forecast_data is None:
            messagebox.showwarning("Warning", "No forecast data to save. Run a forecast first.")
            return

        try:
            db_manager.insert_forecast(
                ticker=self.tkr_var.get().strip().upper(),
                interval=self.interval_var.get(),
                context_length=self.context_len_var.get(),
                horizon_length=self.horizon_var.get(),
                model_repo=self.repo_var.get(),
                period=self.period_var.get(),
                forecast_data=np.asarray(self.forecast_data, dtype=float).reshape(-1).tolist(),
                mae_score=None,  # Filled in later by calculate_mae_for_selected
                target_column=self.forecast_target_col or self.target_col_var.get(),
                anchor_date=self._anchor_to_iso(self.forecast_anchor),
            )
            self.log_message("Forecast results saved to database successfully.")
            self.refresh_forecast_history()
        except Exception as db_e:
            self.log_message(f"Database save failed: {str(db_e)}", "error")

    @staticmethod
    def _anchor_to_iso(anchor):
        """Normalise a forecast anchor to a timezone-naive ISO string."""
        if anchor is None:
            return None
        anchor = pd.Timestamp(anchor)
        if anchor.tzinfo is not None:
            anchor = anchor.tz_localize(None)
        return anchor.isoformat()

    def _actuals_series(self, target_column):
        """Loaded history as a tz-naive Series, or None if unusable."""
        if self.historical_data is None or self.historical_data.empty:
            return None
        if target_column not in self.historical_data.columns:
            return None

        dates = pd.DatetimeIndex(pd.to_datetime(self.historical_data.index))
        if dates.tz is not None:
            dates = dates.tz_localize(None)

        values = np.asarray(self.historical_data[target_column], dtype=float).reshape(-1)
        series = pd.Series(values, index=dates)
        return series[~series.index.duplicated(keep="last")]

    def _on_process_error(self, error_msg):
        self.log_message(f"ERROR: {error_msg}")
        messagebox.showerror("Process Error", str(error_msg))
        self.set_processing_state(False)

    def _get_forecast_dates(self, last_date, interval, length):
        """Return timezone-naive dates immediately after the historical data."""
        last_date = pd.Timestamp(last_date)
        if last_date.tzinfo is not None:
            last_date = last_date.tz_localize(None)

        if interval == "1d":
            return pd.DatetimeIndex(
                pd.bdate_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=length,
                )
            )

        interval_map = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "1wk": pd.Timedelta(weeks=1),
            "1mo": pd.DateOffset(months=1),
        }
        step = interval_map.get(interval, pd.Timedelta(days=1))
        return pd.DatetimeIndex([last_date + (step * offset) for offset in range(1, length + 1)])

    def update_plot(self):
        if not hasattr(self, "ax"):
            return

        self.ax.clear()
        self.ax.set_title(f"{self.tkr_var.get()} Historical & Forecast ({self.target_col_var.get()})")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, linestyle="--", alpha=0.6)

        if self.historical_data is None or self.historical_data.empty:
            self.canvas.draw()
            return

        target = self.target_col_var.get()
        dates = pd.DatetimeIndex(pd.to_datetime(self.historical_data.index))
        if dates.tz is not None:
            dates = dates.tz_localize(None)

        prices = np.asarray(self.historical_data[target], dtype=float).reshape(-1)
        if len(dates) != len(prices):
            raise ValueError("Historical dates and prices have different lengths.")

        self.ax.plot(dates, prices, label="Historical Data", color="blue", linewidth=1.5)
        last_date = dates[-1]
        last_price = float(prices[-1])

        if self.forecast_data is not None:
            forecast_values = np.asarray(self.forecast_data, dtype=float).reshape(-1)
            if forecast_values.size:
                forecast_dates = self._get_forecast_dates(
                    last_date, self.interval_var.get(), forecast_values.size
                )
                self.ax.plot(
                    pd.DatetimeIndex([last_date]).append(forecast_dates),
                    np.concatenate(([last_price], forecast_values)),
                    label="TimesFM Forecast",
                    color="orange",
                    linewidth=2,
                    linestyle="--",
                )

        overlay_colors = ["green", "red", "purple", "brown", "pink", "gray"]
        history = pd.Series(prices, index=dates)
        history = history[~history.index.duplicated(keep="last")]

        for index, record in enumerate(self.overlay_forecasts):
            overlay_values = np.asarray(record["forecast_data"], dtype=float).reshape(-1)
            if not overlay_values.size:
                continue

            # Draw each saved forecast from the point it was actually made.
            # Rows saved before anchor_date existed fall back to the end of the
            # current series, which is what every overlay used to do.
            anchor = record.get("anchor_date")
            if anchor:
                overlay_anchor = pd.Timestamp(anchor)
                if overlay_anchor.tzinfo is not None:
                    overlay_anchor = overlay_anchor.tz_localize(None)
                anchor_price = float(history.get(overlay_anchor, last_price))
                label = f"Saved #{record['id']} @ {overlay_anchor.date()}"
            else:
                overlay_anchor = last_date
                anchor_price = last_price
                label = f"Saved #{record['id']} (unanchored)"

            overlay_dates = self._get_forecast_dates(
                overlay_anchor,
                record.get("interval") or self.interval_var.get(),
                overlay_values.size,
            )
            self.ax.plot(
                pd.DatetimeIndex([overlay_anchor]).append(overlay_dates),
                np.concatenate(([anchor_price], overlay_values)),
                label=label,
                color=overlay_colors[index % len(overlay_colors)],
                linewidth=1.5,
                linestyle=":",
            )

        self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(self.ax.xaxis.get_major_locator()))
        self.ax.yaxis.get_major_formatter().set_useOffset(False)
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

    def refresh_forecast_history(self):
        """Reload the history grid. Safe to call from the UI thread."""
        threading.Thread(target=self._load_forecast_history_job, daemon=True).start()

    def _load_forecast_history_job(self):
        # Tkinter is not thread-safe: query here, touch widgets only via after().
        try:
            history = db_manager.get_forecast_history()
        except Exception as e:
            self.root.after(0, self.log_message,
                            f"Error populating forecast history: {str(e)}", "error")
            return
        self.root.after(0, self._populate_history_grid, history)

    def _populate_history_grid(self, history):
        self.run_tree.delete(*self.run_tree.get_children())

        for record in history:
            mae = record['mae_score']
            self.run_tree.insert("", "end", values=(
                record['id'],
                record['timestamp'],
                record['ticker'],
                record['interval'],
                record['context_length'],
                record['horizon_length'],
                record['model_repo'],
                record['period'],
                record['target_column'] or "-",
                record['anchor_date'] or "-",
                "-" if mae is None else f"{mae:.4f}"
            ))
        self.log_message(f"Forecast history grid updated: {len(history)} record(s) loaded.")

    
    def overlay_selected_forecast(self): 
        selected_rows = self.run_tree.selection()
        if not selected_rows:
            messagebox.showinfo("Overlay Forecast", "Select at least one saved forecast first.")
            return

        try:
            current_ticker = self.tkr_var.get().strip().upper()
            records = []
            for row in selected_rows:
                forecast_id = self.run_tree.item(row, "values")[0]
                record = db_manager.get_forecast_by_id(forecast_id)
                if record is None:
                    continue
                if (record["ticker"] or "").strip().upper() != current_ticker:
                    self.log_message(
                        f"Forecast #{record['id']} is for {record['ticker']}, "
                        f"not {current_ticker}; plotting it anyway.", "warning"
                    )
                records.append(record)

            existing_ids = {record["id"] for record in self.overlay_forecasts}
            new_records = [record for record in records if record["id"] not in existing_ids]
            self.overlay_forecasts.extend(new_records)
            self.update_plot()
            self.log_message(
                f"Overlayed {len(new_records)} new saved forecast(s); "
                f"{len(self.overlay_forecasts)} forecast(s) visible on the plot."
            )
        except Exception as e:
            self.log_message(f"Error retrieving forecast from database: {str(e)}", "error")
            messagebox.showerror("Overlay Forecast", str(e))

    def calculate_mae_for_selected(self):
        """Score saved forecasts against the actuals currently loaded.

        Only points whose forecast date has since materialised in the history
        are compared; a forecast still entirely in the future is skipped.
        """
        selected_rows = self.run_tree.selection()
        if not selected_rows:
            messagebox.showinfo("Calculate MAE", "Select at least one saved forecast first.")
            return

        if self.historical_data is None or self.historical_data.empty:
            messagebox.showwarning(
                "Calculate MAE",
                "Fetch historical data first - actuals are needed to score a forecast."
            )
            return

        current_ticker = self.tkr_var.get().strip().upper()
        scored = 0

        for row in selected_rows:
            forecast_id = self.run_tree.item(row, "values")[0]
            try:
                record = db_manager.get_forecast_by_id(forecast_id)
            except Exception as e:
                self.log_message(f"Could not load forecast #{forecast_id}: {str(e)}", "error")
                continue

            if record is None:
                self.log_message(f"Forecast #{forecast_id} no longer exists.", "warning")
                continue

            if (record["ticker"] or "").strip().upper() != current_ticker:
                self.log_message(
                    f"Skipped #{record['id']}: saved for {record['ticker']}, "
                    f"loaded data is {current_ticker}.", "warning"
                )
                continue

            if not record["anchor_date"]:
                self.log_message(
                    f"Skipped #{record['id']}: saved before anchor dates were "
                    "recorded, so it cannot be aligned to actuals.", "warning"
                )
                continue

            target_column = record["target_column"] or self.target_col_var.get()
            actuals = self._actuals_series(target_column)
            if actuals is None:
                self.log_message(
                    f"Skipped #{record['id']}: column '{target_column}' is not in "
                    "the loaded data.", "warning"
                )
                continue

            forecast_values = np.asarray(record["forecast_data"], dtype=float).reshape(-1)
            if not forecast_values.size:
                self.log_message(f"Skipped #{record['id']}: empty forecast.", "warning")
                continue

            anchor = pd.Timestamp(record["anchor_date"])
            if anchor.tzinfo is not None:
                anchor = anchor.tz_localize(None)

            forecast_dates = self._get_forecast_dates(
                anchor, record["interval"] or self.interval_var.get(), forecast_values.size
            )
            aligned = actuals.reindex(forecast_dates)
            mask = aligned.notna().values

            if not mask.any():
                self.log_message(
                    f"Skipped #{record['id']}: no actuals yet for its forecast window.",
                    "warning"
                )
                continue

            mae = float(np.mean(np.abs(forecast_values[mask] - aligned.values[mask])))

            try:
                db_manager.update_mae_score(record["id"], mae)
            except Exception as e:
                self.log_message(f"Could not save MAE for #{record['id']}: {str(e)}", "error")
                continue

            scored += 1
            self.log_message(
                f"Forecast #{record['id']} ({target_column}): MAE {mae:.4f} over "
                f"{int(mask.sum())}/{forecast_values.size} matched point(s)."
            )

        if scored:
            self.refresh_forecast_history()
        else:
            self.log_message("No selected forecast could be scored.", "warning")

    def delete_selected_forecast(self):
        selected_rows = self.run_tree.selection()
        if not selected_rows:
            messagebox.showinfo("Delete Forecast", "Select at least one saved forecast first.")
            return

        if not messagebox.askyesno(
            "Delete Forecast",
            f"Permanently delete {len(selected_rows)} saved forecast(s)?\n"
            "This cannot be undone."
        ):
            return

        deleted_ids = []
        for row in selected_rows:
            forecast_id = self.run_tree.item(row, "values")[0]
            try:
                if db_manager.delete_forecast(forecast_id):
                    deleted_ids.append(int(forecast_id))
                else:
                    self.log_message(f"Forecast #{forecast_id} was already gone.", "warning")
            except Exception as e:
                self.log_message(f"Delete failed for #{forecast_id}: {str(e)}", "error")

        if deleted_ids:
            # Drop anything we just deleted from the plot as well.
            remaining = [r for r in self.overlay_forecasts if r["id"] not in deleted_ids]
            if len(remaining) != len(self.overlay_forecasts):
                self.overlay_forecasts = remaining
                self.update_plot()

        self.log_message(f"Deleted {len(deleted_ids)} saved forecast(s).")
        self.refresh_forecast_history()

if __name__ == "__main__":
    root = tk.Tk()
    app = TimesFMApp(root)
    root.mainloop()