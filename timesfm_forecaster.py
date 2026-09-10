import tkinter as tk
from tkinter import ttk, messagebox
import threading
import datetime
import traceback
import numpy as np
import pandas as pd
import db_manager
import data_cache
import backtest
import metrics
import transforms
import features
import csv
import os  # Added for scanning local model directories
import sys
# Used to safely build public FRED CSV query URLs for custom economic series.
from urllib.parse import quote

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
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

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


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Weights already sitting in ./models are preferred over the hub id at startup,
# so a first launch with no network still comes up with a usable model.
DEFAULT_REPO = "google/timesfm-3.0-pytorch"
DEFAULT_LOCAL_DIRNAME = "google___timesfm-3.0-pytorch"


def resolve_default_repo():
    local = os.path.join(MODELS_DIR, DEFAULT_LOCAL_DIRNAME)
    return local if os.path.isdir(local) else DEFAULT_REPO


class TimesFMApp:
    def __init__(self, root):
        """Construct the UI, initialize storage, and load saved history asynchronously."""
        self.root = root
        self.root.title("TimesFM 3.0 Stock Data Forecaster")
        self.root.geometry("1400x850")
        self.root.minsize(1000, 700)
        
        # Internal state variables
        self.historical_data = None
        self.forecast_data = None
        self.overlay_forecasts = []
        self.is_processing = False
        
        # Model Caching State to prevent OOM/Bottlenecks
        self.loaded_model = None
        self.current_model_config = {}
        self.validation_forecast_data = None
        self.validation_metrics = []
        self.validation_summary = {}
        self.validation_origin = None
        self.validation_record_saved = False
        # What the live forecast was conditioned on, captured at inference time.
        self.forecast_anchor = None
        self.forecast_target_col = None
        self.forecast_quantiles = None
        self.forecast_space = transforms.PRICE
        self.data_meta = {}
        self.backtest_thread = None
        self.backtest_stop = threading.Event()
        self.model_lock = threading.Lock()
        # These collections describe the fetched feature table and its UI selections.
        self.feature_catalog = []
        self.feature_selected = set()
        self.feature_shown = []
        self.external_source_vars = {}
        
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

        # refresh_forecast_history now queries off-thread and populates on the UI
        # thread. Calling it from a worker touched Tk widgets directly, and the
        # log line read the tree before the worker had filled it, so it always
        # reported 0 records.
        self.refresh_forecast_history()
        self.refresh_backtest_history()

        # Load TimesFM 3.0 in the background so the first forecast does not pay
        # the download and initialisation cost. The UI stays usable throughout.
        threading.Thread(target=self._autoload_model_job, daemon=True).start()

    def setup_ui(self):

        # Creating a parent notebook frame to hold the left and right panels
        parent_notebook = ttk.Notebook(self.root)
        parent_notebook.pack(fill="both", expand=True)

        #Creating the tabs inside the parent notebook
        tab_inference = ttk.Frame(parent_notebook)
        tab_logs = ttk.Frame(parent_notebook)
        tab_settings = ttk.Frame(parent_notebook)

        tab_backtest = ttk.Frame(parent_notebook)

        parent_notebook.add(tab_inference, text="Inference & Visualization")
        parent_notebook.add(tab_logs, text="System Logs")
        parent_notebook.add(tab_backtest, text="Backtest & Validation")
        parent_notebook.add(tab_settings, text="Model Settings")

        # INFERENCE & VISUALIZATION TAB
        # A paned window rather than two packed frames, so the divider between
        # the settings and the plot can be dragged. The settings column carries
        # long labels and file paths, and a fixed 380px forced them behind a
        # scrollbar on every layout.
        self.inference_paned = ttk.PanedWindow(tab_inference, orient=tk.HORIZONTAL)
        self.inference_paned.pack(fill="both", expand=True)

        left_panel = ttk.Frame(self.inference_paned, width=380, padding=(10, 10, 10, 10))
        self.inference_paned.add(left_panel, weight=0)

        # Grid, not pack: a horizontal scrollbar has to sit under the canvas
        # without stealing the vertical one's column. Entries wider than the
        # panel were simply unreachable before, because the canvas only ever
        # had a yscrollcommand.
        left_panel.rowconfigure(0, weight=1)
        left_panel.columnconfigure(0, weight=1)

        canvas = tk.Canvas(left_panel, width=340, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(left_panel, orient="horizontal", command=canvas.xview)
        self.settings_frame = ttk.Frame(canvas)

        self.settings_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        settings_window = canvas.create_window((0, 0), window=self.settings_frame, anchor="nw")

        def fit_settings_width(event):
            """Stretch the settings content to the pane, never below its natural width.

            Clamping at the natural width is what keeps the horizontal scrollbar
            meaningful when the divider is dragged narrow.
            """
            canvas.itemconfigure(
                settings_window,
                width=max(event.width, self.settings_frame.winfo_reqwidth()),
            )

        canvas.bind("<Configure>", fit_settings_width)
        canvas.configure(yscrollcommand=scrollbar.set, xscrollcommand=h_scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        def scroll_settings(event):
            """Scroll the settings canvas when the pointer is over the first tab panel."""
            event_widget = getattr(event, "widget", None)
            while event_widget is not None and event_widget != canvas:
                parent_name = event_widget.winfo_parent()
                if not parent_name:
                    return
                event_widget = event_widget.nametowidget(parent_name)
            if event_widget is None:
                return

            event_num = getattr(event, "num", None)
            event_delta = getattr(event, "delta", 0)
            if event_num == 4:
                scroll_amount = -1
            elif event_num == 5:
                scroll_amount = 1
            elif event_num == 6:
                scroll_amount = -1
            elif event_num == 7:
                scroll_amount = 1
            else:
                scroll_amount = -1 if event_delta > 0 else 1
            canvas.yview_scroll(scroll_amount, "units")
            return "break"

        # Tkinter reports macOS touchpad/wheel gestures as MouseWheel events.
        # Buttons 4-7 are X11 scroll buttons: macOS and Windows Tk reject them
        # outright with 'bad button number', which aborted setup_ui and stopped
        # the app from starting at all. Bind them only where they exist.
        WHEEL_SEQUENCES = ("<MouseWheel>", "<Shift-MouseWheel>", "<Option-MouseWheel>")
        X11_SCROLL_SEQUENCES = ("<Button-4>", "<Button-5>", "<Button-6>", "<Button-7>")

        def bind_wheel(widget, binder):
            for sequence in WHEEL_SEQUENCES:
                binder(widget, sequence)
            if sys.platform.startswith("linux"):
                for sequence in X11_SCROLL_SEQUENCES:
                    try:
                        binder(widget, sequence)
                    except tk.TclError:
                        pass

        bind_wheel(canvas, lambda widget, sequence: widget.bind_all(sequence, scroll_settings))

        def bind_settings_scroll(widget):
            """Give nested settings controls a direct chance to handle touchpad events."""
            # The covariate table has its own scrollbar and handler. Binding the
            # panel handler here too would win, because it is added first and
            # returns "break", leaving the list unscrollable.
            if widget is getattr(self, "feature_listbox", None):
                return
            bind_wheel(widget, lambda w, sequence: w.bind(sequence, scroll_settings, add="+"))
            for child in widget.winfo_children():
                bind_settings_scroll(child)
        


        # LOGS AND QC TAB: tab_logs
        self.data_grid_frame = ttk.LabelFrame(tab_logs, text="Forecast History", padding=(10, 10, 10, 10))
        self.data_grid_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ## DATA GRID and children
        self.run_tree = ttk.Treeview(self.data_grid_frame, columns=("ID", "Timestamp", "Ticker", "Target", "Interval", "Context Length", "Horizon Length", "Model Repo", "Period", "Anchor", "MAE", "MASE", "Dir %"), show="headings", selectmode="extended")
        self.run_tree.pack(fill="both", expand=True, side="left")
        self.run_tree.heading("ID", text="ID",)
        self.run_tree.heading("Timestamp", text="Timestamp")
        self.run_tree.heading("Ticker", text="Ticker")
        self.run_tree.heading("Target", text="Target")
        self.run_tree.heading("Interval", text="Interval")
        self.run_tree.heading("Context Length", text="Context Length")
        self.run_tree.heading("Horizon Length", text="Horizon Length")
        self.run_tree.heading("Model Repo", text="Model Repo")
        self.run_tree.heading("Period", text="Period")
        self.run_tree.heading("Anchor", text="Anchor Date")
        self.run_tree.heading("MAE", text="MAE")
        # MASE is the column to read: < 1 beats a naive forecast.
        self.run_tree.heading("MASE", text="MASE (<1 = skill)")
        self.run_tree.heading("Dir %", text="Dir %")

        tree_scrollbar_horizontal = ttk.Scrollbar(self.data_grid_frame, orient="horizontal", command=self.run_tree.xview)
        tree_scrollbar_vertical = ttk.Scrollbar(self.data_grid_frame, orient="vertical", command=self.run_tree.yview)
        tree_scrollbar_horizontal.pack(side="bottom", fill="x")
        tree_scrollbar_vertical.pack(side="right", fill="y")
        self.run_tree.configure(xscrollcommand=tree_scrollbar_horizontal.set, yscrollcommand=tree_scrollbar_vertical.set)

        ### Configure treeview columns
        self.qc_controls_frame = ttk.LabelFrame(tab_logs, text="QC Controls")
        self.qc_controls_frame.pack(fill="x", pady=(0, 10), padx=10)

        # TAB 3: MODEL MANAGER (THE ARMORY)
        armory_frame = ttk.LabelFrame(tab_settings, text="Local Model Weights", padding=10)
        armory_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.model_listbox = tk.Listbox(armory_frame, font=("Courier", 10))
        self.model_listbox.pack(fill="both", expand=True, pady=(0, 10))

        armory_btn_frame = ttk.Frame(armory_frame)
        armory_btn_frame.pack(fill="x")

        self.refresh_models_btn = ttk.Button(armory_btn_frame, text="Scan Local Folder", command=self.scan_local_models)
        self.refresh_models_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.set_active_btn = ttk.Button(armory_btn_frame, text="Set Selected as Active Model", command=self.set_active_local_model)
        self.set_active_btn.pack(side=tk.LEFT, padx=5)

        download_frame = ttk.LabelFrame(tab_settings, text="Download New Model (HuggingFace)", padding=10)
        download_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(download_frame, text="HF Repo ID:").pack(side=tk.LEFT)
        self.dl_repo_var = tk.StringVar(value="google/timesfm-1.0-200m-pytorch")
        ttk.Entry(download_frame, textvariable=self.dl_repo_var, width=40).pack(side=tk.LEFT, padx=10)
        
        self.download_btn = ttk.Button(download_frame, text="Download Model", command=self.thread_download_model)
        self.download_btn.pack(side=tk.LEFT)
        
        self.build_data_settings()
        self.build_model_settings()
        self.build_action_buttons()
        bind_settings_scroll(self.settings_frame)
        self.scan_local_models() # Auto-populate listbox on boot
        
        right_panel = ttk.Frame(self.inference_paned, padding=(10, 10, 10, 10))
        # weight=1: dragging the divider gives the extra space to the plot, and
        # resizing the window grows the plot rather than the settings column.
        self.inference_paned.add(right_panel, weight=1)
        right_panel.rowconfigure(0, weight=3) 
        right_panel.rowconfigure(1, weight=1) 
        right_panel.columnconfigure(0, weight=1)
        
        self.plot_frame = ttk.LabelFrame(right_panel, text="Data & Forecast Visualization")
        self.plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        self.model_status_label = ttk.Label(right_panel, text="Model: starting...", foreground="#333333")
        self.model_status_label.grid(row=2, column=0, sticky="w", pady=(4, 0))

        self.log_frame = ttk.LabelFrame(right_panel, text="System Logs & Status")
        self.log_frame.grid(row=1, column=0, sticky="nsew")
        
        self.log_text = tk.Text(self.log_frame, state='disabled', wrap='word', height=8, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 10))
        self.log_text.tag_config("warning", foreground="#ffb000")
        self.log_text.tag_config("error", foreground="#ff5555")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.build_backtest_tab(tab_backtest)
        self.init_plot()

        # sashpos only works once the widget has been mapped and sized.
        self.root.after_idle(self._place_inference_sash)

    # ----------------------------------------------------------------
    # Backtest tab
    # ----------------------------------------------------------------

    def build_backtest_tab(self, parent):
        controls = ttk.LabelFrame(parent, text="Walk-Forward Configuration", padding=(10, 5))
        controls.pack(fill="x", padx=10, pady=(10, 5))

        def spin(label, var, column, width=7):
            ttk.Label(controls, text=label).grid(row=0, column=column * 2, sticky="w", padx=(8, 2))
            ttk.Entry(controls, textvariable=var, width=width).grid(row=0, column=column * 2 + 1, sticky="w")

        self.bt_context_var = tk.IntVar(value=256)
        self.bt_horizon_var = tk.IntVar(value=5)
        self.bt_step_var = tk.IntVar(value=5)
        self.bt_cost_var = tk.DoubleVar(value=10.0)

        spin("Context:", self.bt_context_var, 0)
        spin("Horizon:", self.bt_horizon_var, 1)
        spin("Step:", self.bt_step_var, 2)
        spin("Cost (bps):", self.bt_cost_var, 3)

        ttk.Label(controls, text="Window:").grid(row=0, column=8, sticky="w", padx=(8, 2))
        self.bt_mode_var = tk.StringVar(value="sliding")
        ttk.Combobox(controls, textvariable=self.bt_mode_var, values=["sliding", "expanding"],
                     width=10, state="readonly").grid(row=0, column=9, sticky="w")

        ttk.Label(controls, text="Model in:").grid(row=0, column=10, sticky="w", padx=(8, 2))
        self.bt_space_var = tk.StringVar(value=transforms.SPACE_LABELS[transforms.LOG_RETURN])
        ttk.Combobox(controls, textvariable=self.bt_space_var,
                     values=[transforms.SPACE_LABELS[space] for space in transforms.TARGET_SPACES],
                     width=14, state="readonly").grid(row=0, column=11, sticky="w")

        self.bt_baselines_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Include baselines", variable=self.bt_baselines_var
                        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.bt_timesfm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Include TimesFM (slow: one inference per origin)",
                        variable=self.bt_timesfm_var
                        ).grid(row=1, column=3, columnspan=5, sticky="w", pady=(6, 0))

        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", padx=10)
        self.bt_run_btn = ttk.Button(buttons, text="Run Backtest", command=self.thread_run_backtest)
        self.bt_run_btn.pack(side="left", padx=(0, 5), pady=5)
        self.bt_stop_btn = ttk.Button(buttons, text="Stop", command=self.stop_backtest, state="disabled")
        self.bt_stop_btn.pack(side="left", padx=5, pady=5)
        ttk.Button(buttons, text="Refresh Saved", command=self.refresh_backtest_history).pack(side="left", padx=5)
        ttk.Button(buttons, text="Delete Selected", command=self.delete_selected_backtest).pack(side="left", padx=5)
        ttk.Button(buttons, text="Per-Step Detail", command=self.show_step_detail).pack(side="left", padx=5)

        self.bt_progress = ttk.Label(parent, text="Idle.")
        self.bt_progress.pack(fill="x", padx=10)

        columns = ("ID", "Model", "Origins", "MASE h1", "MASE all", "Skill%",
                   "Skill r%", "Dir%", "Dir p", "DM p h1", "DM p", "CRPS",
                   "Cov80", "IC", "Sharpe", "MaxDD", "DSR")
        results = ttk.LabelFrame(parent, text="Results  (MASE < 1 beats naive; DSR is the "
                                              "probability the Sharpe survives how many configs you tried)",
                                 padding=(10, 5))
        results.pack(fill="both", expand=True, padx=10, pady=10)

        self.bt_tree = ttk.Treeview(results, columns=columns, show="headings", selectmode="browse")
        for column in columns:
            self.bt_tree.heading(column, text=column)
            self.bt_tree.column(column, width=72, anchor="center")
        self.bt_tree.column("Model", width=130, anchor="w")
        self.bt_tree.pack(fill="both", expand=True, side="left")

        scroll = ttk.Scrollbar(results, orient="vertical", command=self.bt_tree.yview)
        scroll.pack(side="right", fill="y")
        self.bt_tree.configure(yscrollcommand=scroll.set)

    def stop_backtest(self):
        self.backtest_stop.set()
        self.log_message("Stop requested; finishing the current origin.", "warning")

    def thread_run_backtest(self):
        if self.backtest_thread is not None and self.backtest_thread.is_alive():
            messagebox.showinfo("Backtest", "A backtest is already running.")
            return
        if self.historical_data is None or self.historical_data.empty:
            messagebox.showwarning("Backtest", "Fetch data first.")
            return
        if self.bt_timesfm_var.get() and self.loaded_model is None:
            messagebox.showwarning(
                "Backtest",
                "Run a forecast once first so the model is loaded, or untick TimesFM."
            )
            return
        if self.adjusted_var.get():
            proceed = messagebox.askyesno(
                "Backtest on adjusted prices?",
                "The loaded data is split/dividend adjusted, which folds information "
                "from after each bar into that bar. That is look-ahead bias.\n\n"
                "Untick 'Adjusted prices', refetch, and backtest on raw prices for a "
                "clean result.\n\nRun anyway?"
            )
            if not proceed:
                return

        self.backtest_stop.clear()
        self.bt_run_btn.state(['disabled'])
        self.bt_stop_btn.state(['!disabled'])
        self.backtest_thread = threading.Thread(target=self._run_backtest_job, daemon=True)
        self.backtest_thread.start()

    def _selected_backtest_space(self):
        label = self.bt_space_var.get()
        for space, text in transforms.SPACE_LABELS.items():
            if text == label:
                return space
        return transforms.PRICE

    def _timesfm_model_fn(self, horizon):
        """Wrap the already-loaded model in the backtest's model_fn contract.

        Declares wants_covariates, so the engine hands over a covariate window
        re-sliced at each origin. Standardisation happens here against that
        window only, matching what inference does and keeping it leak-free.
        """
        def _forecast(context, steps, covariate_window=None):
            covariate_array = None
            if covariate_window is not None and covariate_window.size:
                covariate_array = features.standardize_context(covariate_window)
            point, spread = self._predict_loaded_model(
                np.asarray(context, dtype=np.float32), covariate_array, steps,
                want_quantiles=True
            )
            return point, (np.asarray(spread["values"]) if spread else None)
        _forecast.wants_covariates = True
        return _forecast

    def _backtest_covariate_matrix(self, selected):
        """(n_features, n_observations) aligned to the target, forward-filled only."""
        if not selected:
            return None, []
        rows, used = [], []
        for column in selected:
            if column not in self.historical_data.columns:
                continue
            values = pd.to_numeric(self.historical_data[column], errors="coerce")
            if not np.isfinite(values.to_numpy(float)).any():
                continue
            rows.append(values.ffill().to_numpy(float))
            used.append(column)
        return (np.asarray(rows, dtype=float) if rows else None), used

    def _run_backtest_job(self):
        try:
            target = self.target_col_var.get()
            series = self._actuals_series(target)
            if series is None:
                raise ValueError(f"Column '{target}' is not in the loaded data.")

            values = series.to_numpy(float)
            dates = series.index
            context_len = int(self.bt_context_var.get())
            horizon = int(self.bt_horizon_var.get())
            step = max(int(self.bt_step_var.get()), 1)
            mode = self.bt_mode_var.get()
            cost_bps = float(self.bt_cost_var.get())
            interval = self.interval_var.get()
            ticker = self.tkr_var.get().strip().upper()

            origins = backtest.plan_origins(values.size, context_len, horizon, step)
            if not origins:
                raise ValueError(
                    f"{values.size} bars cannot support a {context_len}-bar context "
                    f"plus a {horizon}-step horizon. Fetch a longer period or shrink the context."
                )

            space = self._selected_backtest_space()

            models = {}
            if self.bt_baselines_var.get():
                models.update(backtest.BASELINES)
                extra = backtest.try_statsforecast_baselines()
                if extra:
                    models.update(extra)
                    self.root.after(0, self.log_message,
                                    f"statsforecast found: added {', '.join(extra)}.")
            covariate_matrix, covariates_used = (None, [])
            if self.bt_timesfm_var.get():
                models["timesfm"] = self._timesfm_model_fn(horizon)
                covariate_matrix, covariates_used = self._backtest_covariate_matrix(
                    self._selected_covariates()
                )
                if covariates_used:
                    self.root.after(0, self.log_message,
                                    f"Backtesting with {len(covariates_used)} covariate(s), "
                                    "re-sliced at every origin.")
            if not models:
                raise ValueError("Select at least one model to backtest.")

            if space != transforms.PRICE:
                # Wrapping changes what a baseline means: `naive` on returns
                # predicts the last return again, which is momentum. Name them
                # accordingly so the results table does not mislead.
                models = {f"{name}@{space}": backtest.in_return_space(fn, space)
                          for name, fn in models.items()}
                # Keep an unwrapped price-space naive as the anchor of the table.
                models["naive@price"] = backtest.naive
                self.root.after(0, self.log_message,
                                f"Modelling {transforms.SPACE_LABELS[space].lower()}; "
                                "forecasts are reconstructed to prices before scoring.")

            self.root.after(0, self.log_message,
                            f"Walk-forward: {len(origins)} origins x {len(models)} model(s), "
                            f"horizon {horizon}, step {step}, {mode} window.")

            n_trials = max(db_manager.count_forecast_runs(), 1)
            table, saved = {}, []

            for index, (name, model_fn) in enumerate(models.items(), start=1):
                if self.backtest_stop.is_set():
                    break

                def progress(done, total, label=name, position=index, count=len(models)):
                    self.root.after(0, self._set_backtest_progress,
                                    f"[{position}/{count}] {label}: origin {done}/{total}")

                rows = backtest.walk_forward(
                    values, dates, model_fn, context_len, horizon, step,
                    mode=mode, progress_cb=progress,
                    should_stop=self.backtest_stop.is_set,
                    covariates=covariate_matrix,
                )
                if not rows:
                    continue

                summary = backtest.summarize(rows, interval=interval,
                                             cost_bps=cost_bps, n_trials=n_trials)
                table[name] = summary

                run_id = db_manager.insert_backtest_run(
                    ticker=ticker, interval=interval, period=self.period_var.get(),
                    model_name=name, context_length=context_len, horizon_length=horizon,
                    step=step, mode=mode, adjusted=self.adjusted_var.get(),
                    cost_bps=cost_bps, n_origins=summary.get("n_origins"),
                    n_points=summary.get("n_points"), summary=summary,
                )
                db_manager.insert_backtest_points(run_id, rows)
                saved.append((run_id, name))

            self.root.after(0, self._on_backtest_done, table, saved)

        except Exception as e:
            self.root.after(0, self._on_backtest_error,
                            f"{str(e)}\n{traceback.format_exc()}")

    def _set_backtest_progress(self, text):
        self.bt_progress.config(text=text)

    def _on_backtest_error(self, message):
        self.log_message(str(message), "error")
        messagebox.showerror("Backtest", str(message).strip().splitlines()[0])
        self._finish_backtest()

    def _finish_backtest(self):
        self.bt_run_btn.state(['!disabled'])
        self.bt_stop_btn.state(['disabled'])

    def _on_backtest_done(self, table, saved):
        self._finish_backtest()
        self.bt_progress.config(text=f"Done. {len(saved)} run(s) saved.")
        self.refresh_backtest_history()

        if not table:
            self.log_message("Backtest produced no results.", "warning")
            return

        ranked = sorted(table.items(),
                        key=lambda kv: (kv[1].get("mase_step1") or float('inf')))
        best, best_summary = ranked[0]
        self.log_message(
            f"Best one-step MASE: {best} at "
            f"{self._fmt(best_summary.get('mase_step1'))}."
        )

        # Return-space skill is the cleaner verdict: it is not distorted by
        # drift, and its naive reference is exactly zero.
        by_return_skill = [(name, summary.get("skill_returns"))
                           for name, summary in table.items()
                           if summary.get("skill_returns") is not None
                           and summary.get("skill_returns") == summary.get("skill_returns")]
        if by_return_skill:
            name, skill = max(by_return_skill, key=lambda kv: kv[1])
            self.log_message(
                f"Best return-space skill: {name} at {skill * 100:+.2f}% versus "
                "predicting a zero return."
            )
            if skill <= 0:
                self.log_message(
                    "Nothing beat a zero-return forecast. On price series that is "
                    "the expected result, and it is the honest answer.", "warning"
                )

    def refresh_backtest_history(self):
        threading.Thread(target=self._load_backtest_history_job, daemon=True).start()

    def _load_backtest_history_job(self):
        try:
            runs = db_manager.get_backtest_runs()
        except Exception as e:
            self.root.after(0, self.log_message, f"Could not load backtests: {str(e)}", "error")
            return
        self.root.after(0, self._populate_backtest_grid, runs)

    @staticmethod
    def _fmt(value, spec=".3f", scale=1.0, suffix=""):
        if value is None:
            return "-"
        try:
            number = float(value) * scale
        except (TypeError, ValueError):
            return "-"
        return "-" if number != number else format(number, spec) + suffix

    def _populate_backtest_grid(self, runs):
        self.bt_tree.delete(*self.bt_tree.get_children())
        for run in runs:
            summary = run.get("summary") or {}
            self.bt_tree.insert("", "end", values=(
                run["id"],
                f"{run['model_name']} ({run['ticker']})",
                run.get("n_origins") or "-",
                self._fmt(summary.get("mase_step1")),
                self._fmt(summary.get("mase")),
                self._fmt(summary.get("skill_vs_naive"), ".1f", 100.0, "%"),
                # Return-space skill: drift-free, and the honest read.
                self._fmt(summary.get("skill_returns"), ".1f", 100.0, "%"),
                self._fmt(summary.get("directional_accuracy"), ".0f", 100.0, "%"),
                self._fmt(summary.get("directional_pvalue_step1")),
                self._fmt(summary.get("dm_pvalue_step1")),
                self._fmt(summary.get("dm_pvalue_vs_naive")),
                self._fmt(summary.get("crps"), ".4f"),
                self._fmt(summary.get("coverage_80"), ".0f", 100.0, "%"),
                self._fmt(summary.get("ic")),
                self._fmt(summary.get("sharpe_net"), ".2f"),
                self._fmt(summary.get("max_drawdown"), ".1f", 100.0, "%"),
                self._fmt(summary.get("deflated_sharpe"), ".2f"),
            ))
        self.log_message(f"Backtest grid updated: {len(runs)} run(s).")

    def _selected_backtest_id(self):
        selection = self.bt_tree.selection()
        if not selection:
            messagebox.showinfo("Backtest", "Select a backtest run first.")
            return None
        return int(self.bt_tree.item(selection[0], "values")[0])

    def delete_selected_backtest(self):
        run_id = self._selected_backtest_id()
        if run_id is None:
            return
        if not messagebox.askyesno("Delete Backtest",
                                   f"Delete run #{run_id} and all of its points?"):
            return
        try:
            db_manager.delete_backtest_run(run_id)
            self.log_message(f"Deleted backtest run #{run_id}.")
        except Exception as e:
            self.log_message(f"Delete failed: {str(e)}", "error")
        self.refresh_backtest_history()

    def show_step_detail(self):
        """Error by horizon step. Pooled numbers hide that error grows with h."""
        run_id = self._selected_backtest_id()
        if run_id is None:
            return
        try:
            run = next((r for r in db_manager.get_backtest_runs() if r["id"] == run_id), None)
        except Exception as e:
            self.log_message(f"Could not load run #{run_id}: {str(e)}", "error")
            return
        if run is None:
            return

        by_step = (run.get("summary") or {}).get("by_step") or []
        if not by_step:
            messagebox.showinfo("Per-Step Detail", "This run has no per-step breakdown.")
            return

        self.log_message(f"--- Run #{run_id} ({run['model_name']}) by horizon step ---")
        for entry in by_step:
            # Any of these can be null: nan is stored as JSON null, and a flat
            # forecast has no directional accuracy at all.
            self.log_message(
                f"    h={entry.get('step')}  n={entry.get('n')}  "
                f"MAE={self._fmt(entry.get('mae'), '.4f')}  "
                f"MASE={self._fmt(entry.get('mase'), '.3f')}  "
                f"skill={self._fmt(entry.get('skill_vs_naive'), '+.1f', 100.0, '%')}  "
                f"dir={self._fmt(entry.get('directional_accuracy'), '.0f', 100.0, '%')}"
            )

    def _place_inference_sash(self, position=380):
        """Start the divider where the old fixed-width panel sat."""
        try:
            if self.inference_paned.winfo_width() > position + 120:
                self.inference_paned.sashpos(0, position)
        except tk.TclError:
            pass          # not mapped yet; the default split is fine

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

        features_frame = ttk.LabelFrame(self.settings_frame, text="2. Features and External Series", padding=(10, 5))
        features_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        ttk.Label(features_frame, text="Additional sources (aligned to the market dates):").pack(anchor="w")
        # The full catalogue: policy rates, inflation, FX, energy, metals,
        # credit spreads and risk proxies. Roughly 15 are ticked by default,
        # which is the working range for TimesFM historical covariates.
        self.external_sources = {
            label: (source, symbol)
            for label, (source, symbol, _note) in features.MACRO_CATALOG.items()
        }
        self.external_notes = {
            label: note for label, (_s, _sym, note) in features.MACRO_CATALOG.items()
        }

        preset_row = ttk.Frame(features_frame)
        preset_row.pack(fill=tk.X, pady=(4, 2))
        ttk.Button(preset_row, text="Default 15", width=10,
                   command=lambda: self._apply_macro_preset("default")).pack(side=tk.LEFT)
        ttk.Button(preset_row, text="All", width=5,
                   command=lambda: self._apply_macro_preset("all")).pack(side=tk.LEFT, padx=3)
        ttk.Button(preset_row, text="None", width=6,
                   command=lambda: self._apply_macro_preset("none")).pack(side=tk.LEFT)
        self.macro_count_label = ttk.Label(preset_row, text="")
        self.macro_count_label.pack(side=tk.LEFT, padx=(8, 0))

        # Deliberately NOT a nested scroller. The settings panel is already a
        # scrolling canvas, and putting a second one inside it meant the inner
        # list was clipped by the outer viewport: the wheel scrolled content
        # that had nowhere to appear, so the list read as frozen. One scroller,
        # checkbuttons straight into the panel, and the panel grows to fit.
        for label, (source, symbol) in self.external_sources.items():
            variable = tk.BooleanVar(value=label in features.DEFAULT_MACRO_SELECTION)
            variable.trace_add("write", lambda *_: self._update_macro_count())
            self.external_source_vars[label] = variable
            ttk.Checkbutton(features_frame, text=f"{label}  [{source}:{symbol}]",
                            variable=variable).pack(anchor="w")

        ttk.Label(features_frame, text="Custom sources (yahoo:SYMBOL or fred:SERIES_ID, comma-separated):").pack(anchor="w", pady=(4, 0))
        self.custom_source_var = tk.StringVar()
        ttk.Entry(features_frame, textvariable=self.custom_source_var, width=34).pack(fill=tk.X)
        # Adjusted closes fold in dividends and splits announced AFTER the bar.
        # Fine forward, look-ahead in a backtest, so make it an explicit choice.
        self.adjusted_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(features_frame, text="Adjusted prices (uncheck to backtest)",
                        variable=self.adjusted_var).pack(anchor="w", pady=(6, 0))
        self.force_refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(features_frame, text="Bypass local cache",
                        variable=self.force_refresh_var).pack(anchor="w")

        ttk.Label(features_frame, text="Model covariates (select after fetching):").pack(anchor="w", pady=(6, 0))
        # ~49 rows land here after fetching (OHLCV + 40 engineered columns +
        # macro series), so a fixed 8-row box with no scrollbar showed a sixth
        # of the table with no way to reach the rest.
        feature_tools = ttk.Frame(features_frame)
        feature_tools.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(feature_tools, text="Filter:").pack(side=tk.LEFT)
        self.feature_filter_var = tk.StringVar()
        ttk.Entry(feature_tools, textvariable=self.feature_filter_var, width=12).pack(side=tk.LEFT, padx=(2, 4))
        self.feature_filter_var.trace_add("write", lambda *_: self._render_feature_list())
        ttk.Button(feature_tools, text="All", width=4,
                   command=lambda: self._select_features("all")).pack(side=tk.LEFT)
        ttk.Button(feature_tools, text="None", width=6,
                   command=lambda: self._select_features("none")).pack(side=tk.LEFT, padx=2)
        self.feature_count_label = ttk.Label(feature_tools, text="0 selected")
        self.feature_count_label.pack(side=tk.LEFT, padx=(6, 0))

        feature_holder = ttk.Frame(features_frame)
        feature_holder.pack(fill=tk.X, pady=(2, 0))
        self.feature_listbox = tk.Listbox(feature_holder, height=16,
                                          selectmode=tk.MULTIPLE, exportselection=False)
        feature_scroll = ttk.Scrollbar(feature_holder, orient="vertical",
                                       command=self.feature_listbox.yview)
        self.feature_listbox.configure(yscrollcommand=feature_scroll.set)
        self.feature_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        feature_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Selection lives in a set, not in the widget: filtering repopulates the
        # listbox, and a covariate hidden by the filter must stay selected.
        self.feature_selected = set()
        self.feature_shown = []
        self.feature_listbox.bind("<<ListboxSelect>>", self._on_feature_select)

        def scroll_features(event):
            """Scroll the listbox itself, not the settings panel behind it."""
            number = getattr(event, "num", None)
            if number in (4, 6):
                amount = -1
            elif number in (5, 7):
                amount = 1
            else:
                amount = -1 if getattr(event, "delta", 0) > 0 else 1
            self.feature_listbox.yview_scroll(amount, "units")
            return "break"

        for sequence in ("<MouseWheel>", "<Shift-MouseWheel>", "<Option-MouseWheel>"):
            self.feature_listbox.bind(sequence, scroll_features, add="+")
        if sys.platform.startswith("linux"):
            for sequence in ("<Button-4>", "<Button-5>", "<Button-6>", "<Button-7>"):
                try:
                    self.feature_listbox.bind(sequence, scroll_features, add="+")
                except tk.TclError:
                    pass

    def build_model_settings(self):
        model_frame = ttk.LabelFrame(self.settings_frame, text="2. TimesFM 3.0 Configuration", padding=(10, 5))
        model_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        ttk.Label(model_frame, text="HuggingFace Repo ID:").grid(row=0, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.repo_var = tk.StringVar(value=resolve_default_repo())
        ttk.Entry(model_frame, textvariable=self.repo_var, width=32).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 5))
        ttk.Label(model_frame, text="Context Length (Input):").grid(row=2, column=0, sticky="w", pady=2)
        self.context_len_var = tk.IntVar(value=1056)
        ttk.Entry(model_frame, textvariable=self.context_len_var, width=10).grid(row=2, column=1, sticky="e", pady=2)
        ttk.Label(model_frame, text="Horizon Length (Output):").grid(row=3, column=0, sticky="w", pady=2)
        self.horizon_var = tk.IntVar(value=7)
        ttk.Entry(model_frame, textvariable=self.horizon_var, width=10).grid(row=3, column=1, sticky="e", pady=2)
        ttk.Label(model_frame, text="Validation Ratio (0.05-0.5):").grid(row=4, column=0, sticky="w", pady=2)
        self.validation_ratio_var = tk.DoubleVar(value=0.2)
        ttk.Entry(model_frame, textvariable=self.validation_ratio_var, width=10).grid(row=4, column=1, sticky="e", pady=2)

        # The held-out window is walked, not scored once, so this bounds how
        # many inferences that costs.
        ttk.Label(model_frame, text="Validation origins (max):").grid(row=10, column=0, sticky="w", pady=2)
        self.validation_origins_var = tk.IntVar(value=30)
        ttk.Entry(model_frame, textvariable=self.validation_origins_var, width=10).grid(row=10, column=1, sticky="e", pady=2)

        # Levels make the model track trend and contaminate MASE with drift.
        # Returns are stationary and are what a trade actually depends on.
        ttk.Label(model_frame, text="Model in:").grid(row=9, column=0, sticky="w", pady=2)
        self.target_space_var = tk.StringVar(value=transforms.SPACE_LABELS[transforms.PRICE])
        ttk.Combobox(model_frame, textvariable=self.target_space_var,
                     values=[transforms.SPACE_LABELS[sp] for sp in transforms.TARGET_SPACES],
                     width=14, state="readonly").grid(row=9, column=1, sticky="e", pady=2)
        ttk.Label(model_frame, text="Compute Backend:").grid(row=5, column=0, sticky="w", pady=2)
        self.backend_var = tk.StringVar(value="gpu")
        ttk.Combobox(model_frame, textvariable=self.backend_var, values=["cpu", "gpu", "tpu"], width=8, state="readonly").grid(row=5, column=1, sticky="e", pady=2)
        ttk.Label(model_frame, text="Freq Indicator (0=High, 1=Min...):").grid(row=6, column=0, sticky="w", pady=2)
        self.freq_ind_var = tk.IntVar(value=0)
        ttk.Entry(model_frame, textvariable=self.freq_ind_var, width=10).grid(row=6, column=1, sticky="e", pady=2)

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
        prefix = {"warning": "WARN ", "error": "ERROR "}.get(level, "")
        formatted_msg = f"[{timestamp}] {prefix}{message}\n"

        self.log_text.config(state='normal')
        # The level argument was accepted but ignored, so warnings and errors
        # were indistinguishable from ordinary chatter.
        self.log_text.insert(tk.END, formatted_msg, level if level in ("warning", "error") else ())
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        # No update_idletasks(): this runs inside after() callbacks, and pumping
        # the event loop from one invites re-entrant redraws.

    def set_processing_state(self, state):
        self.is_processing = state
        flag = 'disabled' if state else '!disabled'
        for button in (self.fetch_btn, self.run_btn, self.insert_db_btn, self.save_btn):
            button.state([flag])

    def thread_fetch_data(self):
        if self.is_processing: return
        self.set_processing_state(True)
        self.log_message(f"Fetching data for {self.tkr_var.get()}...")
        threading.Thread(target=self._fetch_data_job, daemon=True).start()

    def _fetch_data_job(self):
        """Fetch raw market data, then add local and external feature series."""
        """Fetch market data, enrich it, and align any requested external series."""
        try:
            ticker = self.tkr_var.get().strip().upper()
            period = self.period_var.get()
            interval = self.interval_var.get()

            # Served locally when fresh. Beyond saving a round trip this is what
            # makes a backtest reproducible: yfinance revises history.
            data, meta = data_cache.get_ohlcv(
                ticker, period, interval,
                adjusted=self.adjusted_var.get(),
                force_refresh=self.force_refresh_var.get(),
            )
            self.data_meta = meta

            if data.empty:
                raise ValueError(f"No data returned for ticker {ticker}.")

            self.historical_data = self._add_engineered_features(data)
            self._fetch_external_series()
            self.root.after(0, self._on_fetch_success)
        except Exception as e:
            self.root.after(0, self._on_process_error, str(e))

    def _on_fetch_success(self):
        self._refresh_feature_controls()
        meta = self.data_meta or {}
        detail = f" (from {meta.get('source', 'download')}, {'adjusted' if meta.get('adjusted') else 'raw'} prices)"
        self.log_message(f"Successfully fetched {len(self.historical_data)} rows and {len(self.historical_data.columns)} usable features{detail}.")
        if not meta.get("adjusted", True):
            self.log_message("Raw prices in use: correct for backtesting, but splits show as jumps.", "warning")
        self.overlay_forecasts.clear()
        # A refetch invalidates a forecast drawn against the previous series.
        if self.forecast_data is not None:
            self.forecast_data = None
            self.forecast_anchor = None
            self.forecast_quantiles = None
            self.log_message("Cleared the previous forecast; re-run it against the new data.", "warning")
        self.update_plot()
        self.set_processing_state(False)

    def _add_engineered_features(self, data):
        """Technical and statistical columns, all computed causally.

        Delegates to features.build_features so the backtest engine and the GUI
        derive covariates the same way.
        """
        return features.build_features(data)

    def _fetch_external_series(self):
        """Fetch the selected macro series and align them to the market calendar.

        Forward fill only. The previous version also back-filled, which writes a
        series' first known value into every earlier date - future information
        sitting in the training window. Leading gaps stay NaN and the rows that
        cannot support the full covariate set are reported instead.
        """
        selected = self._selected_macro_labels()

        custom_sources = [item.strip() for item in self.custom_source_var.get().split(",") if item.strip()]
        for item in custom_sources:
            try:
                source_type, symbol = item.split(":", 1)
                if source_type.lower() not in {"yahoo", "fred"} or not symbol:
                    raise ValueError
                label = f"{source_type.lower()}:{symbol}"
                self.external_sources[label] = (source_type.lower(), symbol)
                selected.append(label)
            except ValueError:
                raise ValueError(f"Invalid custom source '{item}'. Use yahoo:SYMBOL or fred:SERIES_ID.")

        if not selected:
            return

        market_index = pd.DatetimeIndex(pd.to_datetime(self.historical_data.index))
        if market_index.tz is not None:
            market_index = market_index.tz_localize(None)

        failures = []
        for position, label in enumerate(selected, start=1):
            source_type, symbol = self.external_sources[label]
            self.root.after(0, self.log_message,
                            f"Covariate {position}/{len(selected)}: {label} ({source_type}:{symbol})")
            try:
                series = self._download_external(source_type, symbol)
                self.historical_data[label] = features.align_external(
                    series, market_index, label
                ).to_numpy()
            except Exception as error:
                # One bad ticker must not lose the other nineteen.
                failures.append(f"{label} ({str(error)[:60]})")

        if failures:
            self.root.after(0, self.log_message,
                            f"{len(failures)} covariate(s) unavailable and skipped: "
                            + "; ".join(failures), "warning")

        loaded = [label for label in selected if label in self.historical_data.columns]
        if loaded:
            coverage = self.historical_data[loaded].notna().all(axis=1).sum()
            self.root.after(0, self.log_message,
                            f"{len(loaded)} covariate(s) aligned; {coverage} of "
                            f"{len(market_index)} bars have all of them present.")

    def _download_external(self, source_type, symbol):
        """One external series as a float Series indexed by date."""
        if source_type == "yahoo":
            frame = yf.download(symbol, period=self.period_var.get(),
                                interval=self.interval_var.get(),
                                progress=False, timeout=30, auto_adjust=False)
            if frame is None or frame.empty:
                raise ValueError("no data returned")
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)
            column = "Adj Close" if "Adj Close" in frame.columns else "Close"
            return frame[column]

        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote(symbol)}"
        frame = pd.read_csv(url)
        if frame.shape[1] < 2:
            raise ValueError("unexpected FRED response")
        # FRED renamed its date column from DATE to observation_date, and writes
        # missing observations as a bare '.', which would otherwise import as text.
        frame = frame.set_index(frame.columns[0])
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
        if values.empty:
            raise ValueError("no numeric observations")
        return values

    def _apply_macro_preset(self, preset):
        for label, variable in self.external_source_vars.items():
            if preset == "all":
                variable.set(True)
            elif preset == "none":
                variable.set(False)
            else:
                variable.set(label in features.DEFAULT_MACRO_SELECTION)
        self._update_macro_count()

    def _selected_macro_labels(self):
        return [label for label, variable in self.external_source_vars.items() if variable.get()]

    def _update_macro_count(self):
        if hasattr(self, "macro_count_label"):
            count = len(self._selected_macro_labels())
            self.macro_count_label.config(text=f"{count} of {len(self.external_source_vars)} selected")

    def _refresh_feature_controls(self):
        """Refresh target and covariate choices after the feature table changes."""
        target_values = list(self.historical_data.columns)
        self.feature_catalog = [column for column in target_values if column != self.target_col_var.get()]
        # Keep whatever is still available after a refetch.
        self.feature_selected &= set(self.feature_catalog)
        self._render_feature_list()
        for child in self.settings_frame.winfo_children():
            for widget in child.winfo_children():
                if isinstance(widget, ttk.Combobox) and widget.cget("textvariable") == str(self.target_col_var):
                    widget.configure(values=target_values)
        if self.target_col_var.get() not in target_values:
            self.target_col_var.set("Close" if "Close" in target_values else target_values[0])

    def _render_feature_list(self):
        """Show the catalogue filtered by the search box, preserving selection."""
        needle = self.feature_filter_var.get().strip().lower()
        self.feature_shown = [
            column for column in self.feature_catalog if needle in column.lower()
        ]
        self.feature_listbox.delete(0, tk.END)
        for position, column in enumerate(self.feature_shown):
            self.feature_listbox.insert(tk.END, column)
            if column in self.feature_selected:
                self.feature_listbox.selection_set(position)
        self._update_feature_count()

    def _on_feature_select(self, _event=None):
        """Sync the selection set from the rows currently on screen."""
        chosen = set(self.feature_listbox.curselection())
        for position, column in enumerate(self.feature_shown):
            if position in chosen:
                self.feature_selected.add(column)
            else:
                self.feature_selected.discard(column)
        self._update_feature_count()

    def _select_features(self, mode):
        """All/None act on what the filter is showing, not the whole catalogue."""
        for column in self.feature_shown:
            if mode == "all":
                self.feature_selected.add(column)
            else:
                self.feature_selected.discard(column)
        self._render_feature_list()

    def _update_feature_count(self):
        if hasattr(self, "feature_count_label"):
            self.feature_count_label.config(
                text=f"{len(self.feature_selected)} selected"
            )

    def _selected_covariates(self):
        """Return the feature names selected for TimesFM historical covariates."""
        # From the set, so a covariate hidden by the filter is still used.
        return [column for column in self.feature_catalog if column in self.feature_selected]

    def _prepare_context(self, values, end_index, context_len):
        """Return a finite, patch-sized target context ending at ``end_index``."""
        context = np.asarray(values[:end_index], dtype=np.float32)
        if context.size == 0:
            raise ValueError("The validation training partition is empty.")
        if context.size < context_len:
            context = np.pad(context, (context_len - context.size, 0), mode="edge")
        return context[-context_len:]

    def _prepare_covariate_context(self, end_index, context_len, selected_covariates):
        """Historical-only covariates on the target's time axis, standardised.

        Forward fill only. The previous version interpolated with
        limit_direction="both", which reconstructs an earlier value from a later
        one; even inside the context that misrepresents what was knowable at
        each timestamp. Remaining leading gaps are left NaN here and land on the
        context mean once standardised.

        Standardisation matters once there are more than a couple of covariates:
        a policy rate near 50, USDTRY near 40 and a share price near 400 are not
        comparable magnitudes, and without scaling the largest column dominates
        regardless of how much information it carries. Statistics come from the
        context window itself, so this stays leak-free at every origin.
        """
        covariate_context = []
        self.covariate_names_used = []
        for column in selected_covariates:
            values = pd.to_numeric(self.historical_data[column], errors="coerce").to_numpy(dtype=float)
            values = values[:end_index]
            if not np.isfinite(values).any():
                continue
            values = pd.Series(values).ffill().to_numpy()
            if len(values) < context_len:
                values = np.pad(values, (context_len - len(values), 0), mode="edge")
            covariate_context.append(values[-context_len:])
            self.covariate_names_used.append(column)

        if not covariate_context:
            return None
        return features.standardize_context(np.asarray(covariate_context, dtype=np.float32))

    def _model_request_config(self, context_len=None, horizon=None):
        config = {
            "repo": self.repo_var.get(),
            "backend": self.backend_var.get(),
            "mode": EVALUATOR_MODE,
        }
        if EVALUATOR_MODE != "timesfm3":
            # Only the legacy constructor bakes these in.
            config["context_len"] = context_len
            config["horizon"] = horizon
        return config

    def _ensure_model_loaded(self, config, context_len, horizon, input_patch_len=32):
        """Load the model if it is not already loaded for this configuration.

        Guarded by a lock because startup autoload and a manual Run Forecast can
        both reach it, and loading the same weights twice is how you run out of
        memory.
        """
        with self.model_lock:
            if self.loaded_model is not None and self.current_model_config == config:
                return False

            self.root.after(0, self.log_message,
                            f"Loading TimesFM from {config['repo']} (this takes time)...")

            device_target = config["backend"]
            if sys.platform == "darwin" and device_target == "gpu":
                device_target = "mps"
                self.root.after(0, self.log_message, "macOS detected: mapped GPU to Apple MPS.")

            if EVALUATOR_MODE == "timesfm3":
                model_config = ModelConfig(
                    checkpoint_path=config["repo"],
                    device=device_target,
                    per_core_batch_size=1,
                )
                self.loaded_model = TimesFM3Evaluator(model_config)
            else:
                self.loaded_model = timesfm.TimesFm(
                    context_len=context_len,
                    horizon_len=horizon,
                    input_patch_len=input_patch_len,
                    output_patch_len=128,
                    num_layers=20,
                    model_dims=1280,
                    backend=device_target,
                )
                self.loaded_model.load_from_checkpoint(repo_id=config["repo"])

            self.current_model_config = config
            return True

    def _autoload_model_job(self):
        if not TIMESFM_AVAILABLE:
            self.root.after(0, self._set_model_status, "timesfm not installed", "error")
            return
        try:
            self.root.after(0, self._set_model_status, "Loading TimesFM 3.0...", "info")
            config = self._model_request_config(self.context_len_var.get(), self.horizon_var.get())
            self._ensure_model_loaded(config, self.context_len_var.get(), self.horizon_var.get())
            label = os.path.basename(config["repo"].rstrip("/")) or config["repo"]
            self.root.after(0, self._set_model_status, f"Ready: {label}", "info")
            self.root.after(0, self.log_message, "TimesFM 3.0 loaded and ready.")
        except Exception as e:
            # A failed preload must not stop the app: forecasting will retry.
            self.root.after(0, self._set_model_status, "Load failed (will retry on Run)", "error")
            self.root.after(0, self.log_message,
                            f"Model preload failed: {str(e)}. It will be retried on the first forecast.",
                            "warning")

    def _set_model_status(self, text, level="info"):
        colour = {"error": "#b00020", "info": "#0a7d28"}.get(level, "#333333")
        if hasattr(self, "model_status_label"):
            self.model_status_label.config(text=f"Model: {text}", foreground=colour)

    def _predict_loaded_model(self, input_context, covariate_array, horizon,
                              want_quantiles=False):
        """Run one prediction and normalise its output.

        Returns the point forecast, or (point, quantiles) when asked. The
        quantile spread was previously requested as False and discarded; it is
        the most useful thing the model produces.
        """
        input_context = np.asarray(input_context, dtype=np.float32)
        raw_quantiles = None
        if EVALUATOR_MODE == "timesfm3":
            outputs = list(self.loaded_model.predict_batch(
                [input_context],
                horizon=horizon,
                past_only_covariates=[covariate_array],
                return_quantiles=True,
                use_symmetric_averaging=False,
                univariate=False,
            ))
            result = outputs[0].forecast
            raw_quantiles = getattr(outputs[0], "quantiles", None)
        else:
            point_forecast, quantile_forecast = self.loaded_model.forecast(
                [input_context], freq=[self.freq_ind_var.get()]
            )
            result = point_forecast[0]
            # Legacy timesfm returns [batch, horizon, 10]: column 0 is the mean,
            # 1..9 are deciles. This was being thrown away as `_`.
            if quantile_forecast is not None:
                raw_quantiles = np.asarray(quantile_forecast)[0]

        point = np.asarray(result, dtype=float).reshape(-1)[:horizon]
        if not want_quantiles:
            return point
        return point, self._normalise_quantiles(raw_quantiles, horizon)

    def _selected_space(self):
        label = self.target_space_var.get()
        for space, text in transforms.SPACE_LABELS.items():
            if text == label:
                return space
        return transforms.PRICE

    @staticmethod
    def _normalise_quantiles(raw, horizon):
        """Coerce whatever the backend returned into {levels, values}.

        The evaluator paths disagree on shape and neither documents it firmly,
        so anything unrecognised degrades to no quantiles rather than crashing
        mid-forecast.
        """
        if raw is None:
            return None
        try:
            matrix = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return None

        if matrix.ndim == 3 and matrix.shape[0] == 1:
            matrix = matrix[0]
        if matrix.ndim != 2:
            return None
        if matrix.shape[0] != horizon and matrix.shape[1] == horizon:
            matrix = matrix.T
        if matrix.shape[0] != horizon:
            return None

        width = matrix.shape[1]
        if width == 10:
            matrix = matrix[:, 1:]          # column 0 is the mean, not a quantile
            levels = [round(0.1 * i, 1) for i in range(1, 10)]
        elif width == 9:
            levels = [round(0.1 * i, 1) for i in range(1, 10)]
        else:
            levels = [round((i + 1) / (width + 1), 4) for i in range(width)]

        if not np.isfinite(matrix).all():
            return None
        matrix = np.sort(matrix, axis=1)     # quantiles must not cross
        return {"levels": levels, "values": matrix.tolist()}

    @staticmethod
    def _fmt(value, spec=".3f", scale=1.0, suffix=""):
        if value is None:
            return "-"
        try:
            number = float(value) * scale
        except (TypeError, ValueError):
            return "-"
        return "-" if number != number else format(number, spec) + suffix

    def thread_run_forecast(self):
        if self.is_processing: return
        if self.historical_data is None or self.historical_data.empty:
            messagebox.showwarning("Warning", "Please fetch data first.")
            return
            
        self.set_processing_state(True)
        self.log_message("Initializing forecast sequence...")
        threading.Thread(target=self._run_forecast_job, daemon=True).start()

    def _run_forecast_job(self):
        """Prepare the TimesFM input contract and execute the selected model API.

        The target is a one-dimensional context. TimesFM 3 can additionally
        receive historical covariates in ``[feature_count, time_steps]`` form.
        """
        try:
            target_col = self.target_col_var.get()
            if target_col not in self.historical_data.columns:
                raise ValueError(f"Column '{target_col}' not found in fetched data.")
                
            time_series = pd.to_numeric(self.historical_data[target_col], errors="coerce").to_numpy(dtype=float)
            
            if np.isnan(time_series).all():
                raise ValueError(f"The column '{target_col}' contains only invalid/NaN data.")
            
            gaps = int(np.isnan(time_series).sum())
            if gaps:
                # Carry the last observed price across gaps. Imputing the series
                # mean injects prices that never traded near those dates.
                time_series = pd.Series(time_series).ffill().bfill().to_numpy()
                self.root.after(0, self.log_message,
                                f"Filled {gaps} missing value(s) in '{target_col}' by carrying prices forward.",
                                "warning")
            
            context_len = self.context_len_var.get()
            horizon = self.horizon_var.get()
            validation_ratio = float(self.validation_ratio_var.get())
            
            if context_len <= 0:
                raise ValueError("Context length must be strictly greater than 0.")
            if horizon <= 0:
                raise ValueError("Horizon length must be strictly greater than 0.")
            if not 0.05 <= validation_ratio <= 0.5:
                raise ValueError("Validation ratio must be between 0.05 and 0.5.")
                
            # FORCE CONTEXT LENGTH TO BE A MULTIPLE OF INPUT_PATCH_LEN (32) TO PREVENT TENSOR SHAPE MISMATCH
            input_patch_len = 32
            if context_len % input_patch_len != 0:
                old_len = context_len
                context_len = ((context_len // input_patch_len) + 1) * input_patch_len
                self.root.after(0, self.log_message, f"Adjusted Context Length from {old_len} to {context_len} (must be multiple of {input_patch_len}).")
            
            # Do NOT pad the series here. The old code padded time_series up to
            # context_len, which did two bad things: it fed the model a context
            # that was mostly one repeated edge value (1056 requested against 254
            # real bars is 76% fabricated), and it made positions in time_series
            # no longer correspond to rows of historical_data, so the validation
            # split indexed a 254-row DatetimeIndex with 843. Clamp instead, and
            # let _prepare_context pad only the slice it hands to the model.
            if len(time_series) < context_len:
                usable = (len(time_series) // input_patch_len) * input_patch_len
                if usable >= input_patch_len:
                    self.root.after(0, self.log_message,
                                    f"Only {len(time_series)} bars available: context reduced from "
                                    f"{context_len} to {usable}. Fetch a longer period to use more.",
                                    "warning")
                    context_len = usable
                else:
                    self.root.after(0, self.log_message,
                                    f"Only {len(time_series)} bars available, fewer than one "
                                    f"{input_patch_len}-bar patch; the context will be padded.",
                                    "warning")

            selected_covariates = self._selected_covariates()

            # Return space is only safe without covariates: differencing drops
            # the first observation, which would shift the target one bar
            # relative to covariate columns and silently misalign them.
            space = self._selected_space()
            if space != transforms.PRICE and selected_covariates:
                space = transforms.PRICE
                self.root.after(0, self.log_message,
                                "Covariates are selected, so modelling stays in price space "
                                "(differencing would misalign the covariate columns by one bar).",
                                "warning")
            price_series = time_series
            # series_dates[i] is the timestamp of time_series[i]. Differencing
            # drops the first observation, so without this map every position
            # would be off by one against historical_data in return space.
            frame_index = pd.DatetimeIndex(pd.to_datetime(self.historical_data.index))
            row_offset = 0
            if space != transforms.PRICE:
                time_series = transforms.encode(price_series, space)
                row_offset = 1
                self.root.after(0, self.log_message,
                                f"Modelling {transforms.SPACE_LABELS[space].lower()} "
                                f"({time_series.size} observations after differencing).")
            series_dates = frame_index[row_offset:]
            if len(series_dates) != len(time_series):
                raise ValueError(
                    f"Internal alignment error: {len(time_series)} values against "
                    f"{len(series_dates)} dates."
                )

            input_context = self._prepare_context(time_series, len(time_series), context_len)
            covariate_array = self._prepare_covariate_context(
                len(time_series) + row_offset, context_len, selected_covariates
            )
            # TimesFM 3 expects one target context and optional covariates as
            # [feature_count, time_steps]; future values are intentionally unknown.
            
            if not TIMESFM_AVAILABLE:
                raise ImportError("timesfm library is not available. Cannot run forecast.")
                
            # IMPLEMENT MODEL CACHING TO PREVENT OOM CRASHES AND HEAVY BOTTLENECKS
            current_request_config = self._model_request_config(context_len, horizon)

            if not self._ensure_model_loaded(current_request_config, context_len, horizon, input_patch_len):
                self.root.after(0, self.log_message, "Using cached model weights in VRAM/RAM...")
            
            self.root.after(0, self.log_message, "Running inference on prepared context data...")

            # Validation is a walk-forward over the held-out tail, not a single
            # split. Scoring one origin gave `horizon` points - 7 here - no
            # matter how much data was held out: a 0.2 ratio on 500 bars reserved
            # 100 and threw 93 of them away. Seven points cannot separate skill
            # from luck, and a config change swung the headline number from -5%
            # to +28% on exactly that evidence.
            series_length = len(time_series)
            validation_size = max(1, int(np.ceil(series_length * validation_ratio)))
            validation_size = min(validation_size, max(series_length - input_patch_len, 1))
            split_index = series_length - validation_size
            if split_index < 1:
                raise ValueError(
                    f"Only {series_length} observations: not enough to hold out "
                    f"{validation_size} for validation. Fetch a longer period or "
                    "lower the validation ratio."
                )

            # Score in price space whatever the model was fitted on, so the naive
            # reference stays "no change" and the numbers stay comparable.
            validation_values = np.asarray(price_series, dtype=float).reshape(-1)
            validation_dates = frame_index
            validation_model = self._timesfm_model_fn(horizon)
            if space != transforms.PRICE:
                validation_model = backtest.in_return_space(validation_model, space)

            covariate_matrix, covariates_used = self._backtest_covariate_matrix(
                selected_covariates
            )

            first_origin = split_index - 1 + row_offset
            last_origin = len(validation_values) - horizon - 1
            candidates = list(range(first_origin, last_origin + 1))
            if not candidates:
                raise ValueError(
                    f"The held-out window is shorter than the {horizon}-step horizon. "
                    "Lower the horizon or raise the validation ratio."
                )

            max_origins = max(int(self.validation_origins_var.get()), 1)
            origin_step = max(1, int(np.ceil(len(candidates) / max_origins)))
            chosen_origins = candidates[::origin_step]

            self.root.after(0, self.log_message,
                            f"Validating across {len(chosen_origins)} origin(s) of the "
                            f"{validation_size}-bar hold-out (every {origin_step} bar(s)), "
                            f"{horizon} steps each.")

            validation_rows = backtest.walk_forward(
                validation_values, validation_dates, validation_model,
                context_len, horizon, origin_step, mode="sliding",
                origins=chosen_origins, covariates=covariate_matrix,
            )

            summary = backtest.summarize(
                validation_rows, interval=self.interval_var.get(),
                n_trials=max(db_manager.count_forecast_runs(), 1),
            )
            self.validation_summary = summary
            self.validation_rows = validation_rows
            self.validation_forecast_data = np.asarray(
                [row["y_pred"] for row in validation_rows if row["origin_index"] == chosen_origins[-1]],
                dtype=float,
            )
            self.validation_metrics = [
                ("Points", float(summary.get("n_points", 0))),
                ("Origins", float(summary.get("n_origins", 0))),
                ("MAE", summary.get("mae")),
                ("NaiveMAE", summary.get("naive_mae")),
                ("SkillVsNaive", summary.get("skill_vs_naive")),
                ("MASE_h1", summary.get("mase_step1")),
                # Naive's own MASE, as a sanity check: it must sit near 1.0, so
                # a wild MASE_h1 is visibly the model and not the yardstick.
                ("NaiveMASE_h1", summary.get("naive_mase_step1")),
                ("MASE_all", summary.get("mase")),
                ("DirectionalAccuracy", summary.get("directional_accuracy")),
                ("Dir_p_h1", summary.get("directional_pvalue_step1")),
                ("DM_p_h1", summary.get("dm_pvalue_step1")),
                ("DM_p_pooled", summary.get("dm_pvalue_vs_naive")),
            ]
            self.validation_metrics = [
                (name, value) for name, value in self.validation_metrics if value is not None
            ]
            self.validation_origin = pd.Timestamp(validation_dates[chosen_origins[-1]]).isoformat()
            self.validation_record_saved = False

            # Persist it as a backtest run: a walk-forward over a held-out window
            # is exactly that, and it belongs beside the others in the QC tab.
            try:
                run_id = db_manager.insert_backtest_run(
                    ticker=self.tkr_var.get().strip().upper(),
                    interval=self.interval_var.get(), period=self.period_var.get(),
                    model_name=f"timesfm@validation({space})",
                    context_length=context_len, horizon_length=horizon,
                    step=origin_step, mode="sliding",
                    adjusted=self.adjusted_var.get(), cost_bps=0.0,
                    n_origins=summary.get("n_origins"), n_points=summary.get("n_points"),
                    summary=summary,
                )
                db_manager.insert_backtest_points(run_id, validation_rows)
                self.root.after(0, self.refresh_backtest_history)
            except Exception as save_error:
                self.root.after(0, self.log_message,
                                f"Could not store the validation run: {str(save_error)}", "warning")

            self.root.after(
                0, self.log_message,
                "Validation: " + ", ".join(
                    f"{name}={value:.4f}" for name, value in self.validation_metrics
                    if value is not None and value == value
                ),
            )

            # State the conclusion, so a number that cannot support one is not
            # mistaken for evidence.
            skill = summary.get("skill_step1")
            # Matched to the skill it is quoted beside: both are h=1.
            p_value = summary.get("dm_pvalue_step1")
            if skill is not None and skill == skill:
                verdict = "beats naive at h=1" if skill > 0 else "does NOT beat naive at h=1"
                significant = p_value is not None and p_value == p_value and p_value < 0.05
                shown = "n/a" if p_value is None or p_value != p_value else format(p_value, ".3f")
                self.root.after(
                    0, self.log_message,
                    f"Verdict: {verdict} ({skill * 100:+.1f}%); the difference is "
                    + ("significant" if significant else "NOT significant")
                    + f" (DM p={shown}).",
                    "info" if (skill > 0 and significant) else "warning",
                )

            # The real forecast uses the full historical context after validation.
            forecast_result, spread = self._predict_loaded_model(
                input_context, covariate_array, horizon, want_quantiles=True
            )
            if spread:
                self.root.after(0, self.log_message,
                                f"Captured {len(spread['levels'])} quantile levels.")

            # Reconstruct a price path so the plot, the CSV and every saved
            # score stay in price space regardless of what was modelled.
            if space != transforms.PRICE:
                anchor_price = float(price_series[-1])
                forecast_result = transforms.decode(forecast_result, anchor_price, space)
                if spread:
                    matrix = np.asarray(spread["values"], dtype=float)
                    spread = {"levels": spread["levels"], "values": np.column_stack([
                        transforms.decode(matrix[:, i], anchor_price, space)
                        for i in range(matrix.shape[1])
                    ]).tolist()}

            self.forecast_data = forecast_result
            self.forecast_quantiles = spread
            self.forecast_space = space
            self.forecast_anchor = series_dates[-1]
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
            # Persist the feature recipe with the forecast so later evaluation can
            # identify the target series that was actually forecast.
            anchor = self.forecast_anchor
            if anchor is None:
                anchor = self.historical_data.index[-1]
            anchor = pd.Timestamp(anchor)
            if anchor.tzinfo is not None:
                anchor = anchor.tz_localize(None)

            db_manager.insert_forecast(
                ticker=self.tkr_var.get().strip().upper(),
                interval=self.interval_var.get(),
                context_length=self.context_len_var.get(),
                horizon_length=self.horizon_var.get(),
                model_repo=self.repo_var.get(),
                period=self.period_var.get(),
                target_column=self.forecast_target_col or self.target_col_var.get(),
                forecast_data=np.asarray(self.forecast_data, dtype=float).reshape(-1).tolist(),
                mae_score=None,
                anchor_date=anchor.isoformat(),
                quantiles=self.forecast_quantiles,
                feature_columns=getattr(self, "covariate_names_used", None) or self._selected_covariates(),
                target_space=self.forecast_space,
            )
            self.log_message("Forecast results saved to database successfully.")

            # The validation row is no longer persisted: the schema has no
            # forecast_type column to tell it apart from the real forecast, so
            # storing it would produce two indistinguishable rows. Its metrics
            # are reported here instead.
            if self.validation_metrics:
                self.log_message(
                    "Validation (not persisted): " + ", ".join(
                        f"{name}={value:.4f}" for name, value in self.validation_metrics
                        if np.isfinite(value)
                    ), "warning"
                )
            used = getattr(self, "covariate_names_used", None) or self._selected_covariates()
            if used:
                self.log_message(f"Recorded {len(used)} covariate(s) with the run: {', '.join(used[:6])}"
                                 + (" ..." if len(used) > 6 else ""))
            self.refresh_forecast_history()
        except Exception as db_e:
            self.log_message(f"Database save failed: {str(db_e)}", "error")

    def _on_process_error(self, error_msg):
        self.log_message(f"ERROR: {error_msg}")
        messagebox.showerror("Process Error", str(error_msg))
        self.set_processing_state(False)

    def _get_forecast_dates(self, last_date, interval, length):
        """Return timezone-naive dates immediately after the historical data.

        Daily predictions use business days; intraday, weekly, and monthly
        predictions use the interval-specific offsets defined below.
        """
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
            # Raising escaped into Tk's callback machinery, reached only stderr,
            # and left the user with a silently stale plot.
            self.log_message(f"Cannot plot: {len(dates)} dates against {len(prices)} prices.", "error")
            self.canvas.draw()
            return

        self.ax.plot(dates, prices, label="Historical Data", color="blue", linewidth=1.5)
        last_date = dates[-1]
        last_price = float(prices[-1])

        if self.forecast_data is not None:
            forecast_values = np.asarray(self.forecast_data, dtype=float).reshape(-1)
            if forecast_values.size:
                forecast_dates = self._get_forecast_dates(
                    last_date, self.interval_var.get(), forecast_values.size
                )
                # Draw the distribution first so the line sits on its own band.
                self._draw_quantile_band(forecast_dates)
                self.ax.plot(
                    pd.DatetimeIndex([last_date]).append(forecast_dates),
                    np.concatenate(([last_price], forecast_values)),
                    label="TimesFM Forecast",
                    color="orange",
                    linewidth=2,
                    linestyle="--",
                )

        overlay_colors = ["green", "red", "purple", "brown", "pink", "gray"]
        for index, record in enumerate(self.overlay_forecasts):
            overlay_values = np.asarray(record["forecast_data"], dtype=float).reshape(-1)
            if not overlay_values.size:
                continue

            overlay_dates = self._get_forecast_dates(
                last_date,
                record.get("interval", self.interval_var.get()),
                overlay_values.size,
            )
            self.ax.plot(
                pd.DatetimeIndex([last_date]).append(overlay_dates),
                np.concatenate(([last_price], overlay_values)),
                label=f"Saved Forecast #{record['id']}",
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

    def _actuals_series(self, target_column):
        """Loaded history as a tz-naive Series, or None if unusable."""
        if self.historical_data is None or self.historical_data.empty:
            return None
        if target_column not in self.historical_data.columns:
            return None
        dates = pd.DatetimeIndex(pd.to_datetime(self.historical_data.index))
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        values = pd.to_numeric(self.historical_data[target_column], errors="coerce").to_numpy(float)
        series = pd.Series(values, index=dates)
        return series[~series.index.duplicated(keep="last")]

    def _draw_quantile_band(self, forecast_dates):
        """Shade the predicted quantile spread as a fan.

        A single line implies a certainty the model never claimed.
        """
        spread = self.forecast_quantiles
        if not spread:
            return
        levels = [float(level) for level in spread["levels"]]
        matrix = np.asarray(spread["values"], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(forecast_dates):
            return
        for low, high, alpha in ((0.1, 0.9, 0.12), (0.2, 0.8, 0.15), (0.3, 0.7, 0.18)):
            if low in levels and high in levels:
                self.ax.fill_between(
                    forecast_dates,
                    matrix[:, levels.index(low)], matrix[:, levels.index(high)],
                    color="orange", alpha=alpha, linewidth=0,
                    label=f"{int((high - low) * 100)}% interval",
                )

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
                for index, value in enumerate(self.forecast_data):
                    writer.writerow(["Forecast", index + 1, value])
            self.log_message(f"Forecast successfully exported to {filepath}")
        except Exception as error:
            self.log_message(f"Export Error: {str(error)}")

    def refresh_forecast_history(self):
        """Reload the history grid. Safe to call from the UI thread."""
        threading.Thread(target=self._load_forecast_history_job, daemon=True).start()

    def _load_forecast_history_job(self):
        # Tkinter is not thread-safe: query here, touch widgets only via after().
        try:
            history = db_manager.get_forecast_history()
        except Exception as error:
            self.root.after(0, self.log_message,
                            f"Error populating forecast history: {str(error)}", "error")
            return
        self.root.after(0, self._populate_history_grid, history)

    def _populate_history_grid(self, history):
        self.run_tree.delete(*self.run_tree.get_children())
        for record in history:
            self.run_tree.insert("", "end", values=(
                record['id'], record['timestamp'], record['ticker'],
                record.get('target_column') or "-", record['interval'],
                record['context_length'], record['horizon_length'],
                record['model_repo'], record['period'],
                record.get('anchor_date') or "-",
                self._fmt(record.get('mae_score'), ".4f"),
                self._fmt(record.get('mase_score')),
                self._fmt(record.get('directional_accuracy'), ".0f", 100.0, "%"),
            ))
        self.log_message(f"Forecast history grid updated: {len(history)} record(s) loaded.")

    def overlay_selected_forecast(self):
        selected_rows = self.run_tree.selection()
        if not selected_rows:
            messagebox.showinfo("Overlay Forecast", "Select at least one saved forecast first.")
            return

        try:
            records = []
            for row in selected_rows:
                forecast_id = self.run_tree.item(row, "values")[0]
                record = db_manager.get_forecast_by_id(forecast_id)
                if record is not None:
                    records.append(record)

            existing_ids = {record["id"] for record in self.overlay_forecasts}
            new_records = [record for record in records if record["id"] not in existing_ids]
            self.overlay_forecasts.extend(new_records)
            self.update_plot()
            self.log_message(
                f"Overlayed {len(new_records)} new saved forecast(s); "
                f"{len(self.overlay_forecasts)} forecast(s) visible on the plot."
            )
        except Exception as error:
            self.log_message(f"Error retrieving forecast from database: {str(error)}", "error")
            messagebox.showerror("Overlay Forecast", str(error))

    def calculate_mae_for_selected(self):
        """Persist MAE only for forecast timestamps with finite observations."""
        selected_rows = self.run_tree.selection()
        if not selected_rows:
            messagebox.showinfo("Calculate MAE", "Select at least one saved forecast first.")
            return
        if self.historical_data is None or self.historical_data.empty:
            messagebox.showwarning("Calculate MAE", "Fetch historical data before calculating MAE.")
            return

        actual_dates = pd.DatetimeIndex(pd.to_datetime(self.historical_data.index))
        if actual_dates.tz is not None:
            actual_dates = actual_dates.tz_localize(None)
        calculated = []
        unavailable = []
        unavailable_reasons = {}

        try:
            for row in selected_rows:
                forecast_id = self.run_tree.item(row, "values")[0]
                record = db_manager.get_forecast_by_id(forecast_id)
                if record is None:
                    unavailable.append(str(forecast_id))
                    unavailable_reasons[str(forecast_id)] = "the saved forecast could not be loaded"
                    continue
                if not record.get("anchor_date"):
                    unavailable.append(str(forecast_id))
                    unavailable_reasons[str(forecast_id)] = "it has no saved forecast origin"
                    continue
                if record.get("ticker") and record["ticker"].upper() != self.tkr_var.get().strip().upper():
                    unavailable.append(str(forecast_id))
                    unavailable_reasons[str(forecast_id)] = f"it belongs to ticker {record['ticker']}"
                    continue

                target = record.get("target_column") or self.target_col_var.get()
                if target not in self.historical_data.columns:
                    unavailable.append(str(forecast_id))
                    unavailable_reasons[str(forecast_id)] = f"target column '{target}' is not loaded"
                    continue
                actual_values = pd.to_numeric(self.historical_data[target], errors="coerce").to_numpy(dtype=float)
                actual_series = pd.Series(actual_values, index=actual_dates)

                forecast_values = np.asarray(record["forecast_data"], dtype=float).reshape(-1)
                origin = pd.Timestamp(record["anchor_date"])
                if origin.tzinfo is not None:
                    origin = origin.tz_localize(None)
                forecast_dates = self._get_forecast_dates(
                    origin, record["interval"], forecast_values.size
                )
                observed_values = actual_series.reindex(forecast_dates).to_numpy()
                valid = np.isfinite(forecast_values) & np.isfinite(observed_values)
                if not valid.any():
                    unavailable.append(str(forecast_id))
                    unavailable_reasons[str(forecast_id)] = (
                        f"no observations exist after {origin.date()} for its {record['interval']} interval"
                    )
                    continue

                y_true = observed_values[valid]
                y_pred = forecast_values[valid]
                mae = metrics.mae(y_true, y_pred)

                # MASE scale comes from the context preceding the anchor, so it
                # never sees data the forecast could not have seen.
                context = actual_series[actual_series.index <= origin].to_numpy(float)
                mase = metrics.mase(y_true, y_pred, context, seasonality=1)
                anchor_value = float(actual_series.get(origin, np.nan))
                direction = metrics.directional_accuracy(y_true, y_pred, anchor_value)

                db_manager.update_scores(
                    record["id"], mae_score=mae,
                    mase_score=None if np.isnan(mase) else mase,
                    directional_accuracy=None if np.isnan(direction) else direction,
                )
                calculated.append((record["id"], mae, mase, int(valid.sum()), forecast_values.size))

            self.refresh_forecast_history()
            if calculated:
                summary = ", ".join(
                    f"#{forecast_id}: MAE {mae:.4f} / MASE {mase:.3f} ({matched}/{total} points)"
                    for forecast_id, mae, mase, matched, total in calculated
                )
                self.log_message(f"Scored: {summary}")
                if any(m >= 1.0 for _, _, m, _, _ in calculated if m == m):
                    self.log_message(
                        "MASE >= 1 means the forecast did not beat a naive one.", "warning")
            if unavailable:
                reason_text = "\n".join(
                    f"#{forecast_id}: {unavailable_reasons.get(forecast_id, 'no matching observations')}"
                    for forecast_id in unavailable
                )
                self.log_message(f"MAE unavailable:\n{reason_text}", "warning")
                messagebox.showinfo(
                    "Calculate MAE",
                    "MAE could not be calculated for these forecast(s):\n" + reason_text,
                )
        except Exception as error:
            self.log_message(f"Error calculating MAE: {str(error)}", "error")
            messagebox.showerror("Calculate MAE", str(error))
         
    def delete_selected_forecast(self):
        selected_rows = self.run_tree.selection()
        if not selected_rows:
            messagebox.showinfo("Delete Forecast", "Select at least one saved forecast to delete.")
            return

        if messagebox.askyesno("Confirm Deletion", f"Are you sure you want to permanently delete {len(selected_rows)} forecast(s)?"):
            try:
                for row in selected_rows:
                    forecast_id = self.run_tree.item(row, "values")[0]
                    db_manager.delete_forecast(forecast_id)
                self.refresh_forecast_history()
                self.log_message(f"Successfully deleted {len(selected_rows)} forecast(s) from the database.")
            except Exception as e:
                self.log_message(f"Error deleting forecast: {str(e)}", "error")
                messagebox.showerror("Delete Error", f"Could not delete: {str(e)}")

    def scan_local_models(self):
        self.model_listbox.delete(0, tk.END)
        models_dir = MODELS_DIR
        os.makedirs(models_dir, exist_ok=True)
        
        found_models = False
        for item in os.listdir(models_dir):
            if os.path.isdir(os.path.join(models_dir, item)):
                self.model_listbox.insert(tk.END, item)
                found_models = True
                
        if not found_models:
            self.model_listbox.insert(tk.END, "(No local models found in ./models)")

    def set_active_local_model(self):
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showinfo("Set Active Model", "Select a downloaded model from the list first.")
            return
            
        model_folder = self.model_listbox.get(selection[0])
        if model_folder.startswith("("): return # Ignore the "None found" placeholder
        
        full_path = os.path.abspath(os.path.join(MODELS_DIR, model_folder))
        self.repo_var.set(full_path)
        self.log_message(f"Active model set to local path: {full_path}")
        messagebox.showinfo("Model Updated", f"App will now load weights from:\n{full_path}")

    def thread_download_model(self):
        if not HF_HUB_AVAILABLE:
            messagebox.showerror("Missing Library", "The 'huggingface_hub' library is not installed.\nPlease run: pip install huggingface_hub")
            return
            
        repo_id = self.dl_repo_var.get().strip()
        if not repo_id: return
        
        self.download_btn.state(['disabled'])
        self.log_message(f"Starting download of HuggingFace repo '{repo_id}' to local storage...")
        threading.Thread(target=self._download_model_job, args=(repo_id,), daemon=True).start()

    def _download_model_job(self, repo_id):
        try:
            # Replace slashes with ___ to make it a safe folder name
            folder_name = repo_id.replace("/", "___")
            save_path = os.path.join("./models", folder_name)
            
            # This handles downloading only the necessary model weights
            snapshot_download(repo_id=repo_id, local_dir=save_path)
            
            self.root.after(0, self.log_message, f"Successfully downloaded and verified {repo_id}")
            self.root.after(0, self.scan_local_models)
        except Exception as e:
            self.root.after(0, self.log_message, f"Model download failed: {str(e)}", "error")
        finally:
            self.root.after(0, lambda: self.download_btn.state(['!disabled']))

if __name__ == "__main__":
    root = tk.Tk()
    app = TimesFMApp(root)
    root.mainloop()