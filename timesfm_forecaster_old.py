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
import csv
import sys

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
        self.forecast_quantiles = None
        self.forecast_space = transforms.PRICE
        self.data_meta = {}
        self.backtest_thread = None
        self.backtest_stop = threading.Event()
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
        self.refresh_backtest_history()

    def setup_ui(self):

        # Creating a parent notebook frame to hold the left and right panels
        parent_notebook = ttk.Notebook(self.root)
        parent_notebook.pack(fill="both", expand=True)

        #Creating the tabs inside the parent notebook
        tab_inference = ttk.Frame(parent_notebook)
        tab_logs = ttk.Frame(parent_notebook)
        tab_backtest = ttk.Frame(parent_notebook)

        parent_notebook.add(tab_inference, text="Inference & Visualization")
        parent_notebook.add(tab_logs, text="System Logs")
        parent_notebook.add(tab_backtest, text="Backtest & Validation")

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
        self.run_tree = ttk.Treeview(self.data_grid_frame, columns=("ID", "Timestamp", "Ticker", "Interval", "Context Length", "Horizon Length", "Model Repo", "Period", "Target", "Anchor Date", "MAE", "MASE", "Dir %"), show="headings", selectmode="extended")
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
        self.run_tree.heading("MAE", text="MAE")
        # MASE is the column to read: < 1 beats a naive forecast, >= 1 does not.
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
        self.log_text.tag_config("warning", foreground="#ffb000")
        self.log_text.tag_config("error", foreground="#ff5555")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.build_backtest_tab(tab_backtest)
        self.init_plot()

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
                   "Skill r%", "Dir%", "DM p", "CRPS", "Cov80", "IC", "Sharpe",
                   "MaxDD", "DSR")
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
        """Wrap the already-loaded model in the backtest's model_fn contract."""
        def _forecast(context, steps):
            window = np.asarray(context, dtype=np.float32).reshape(-1)
            if EVALUATOR_MODE == "timesfm3":
                outputs = list(self.loaded_model.predict_batch(
                    [window], horizon=steps, return_quantiles=True,
                    use_symmetric_averaging=False))
                point = np.asarray(outputs[0].forecast, dtype=float).reshape(-1)
                raw = getattr(outputs[0], "quantiles", None)
            else:
                point_forecast, quantile_forecast = self.loaded_model.forecast(
                    [window], freq=[self.freq_ind_var.get()])
                point = np.asarray(point_forecast[0], dtype=float).reshape(-1)
                raw = np.asarray(quantile_forecast)[0] if quantile_forecast is not None else None

            spread = self._normalise_quantiles(raw, steps)
            return point[:steps], (np.asarray(spread["values"]) if spread else None)
        return _forecast

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
            if self.bt_timesfm_var.get():
                models["timesfm"] = self._timesfm_model_fn(horizon)
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
        ttk.Combobox(data_frame, textvariable=self.target_col_var, values=["Open", "High", "Low", "Close", "Adj Close", "Volume"], width=13, state="readonly").grid(row=3, column=1, sticky="w", pady=2)

        # Adjusted closes embed dividends and splits announced AFTER the bar.
        # Fine looking forward, look-ahead in a backtest, so make it a choice.
        self.adjusted_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(data_frame, text="Adjusted prices (uncheck to backtest)",
                        variable=self.adjusted_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=2)

        self.force_refresh_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(data_frame, text="Bypass local cache",
                        variable=self.force_refresh_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=2)

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

        # Modelling levels makes the model track the trend and contaminates
        # MASE with drift. Returns are stationary and are what a trade depends on.
        ttk.Label(model_frame, text="Model in:").grid(row=6, column=0, sticky="w", pady=2)
        self.target_space_var = tk.StringVar(value=transforms.SPACE_LABELS[transforms.LOG_RETURN])
        ttk.Combobox(model_frame, textvariable=self.target_space_var,
                     values=[transforms.SPACE_LABELS[space] for space in transforms.TARGET_SPACES],
                     width=14, state="readonly").grid(row=6, column=1, sticky="e", pady=2)

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
        # Levels were accepted but ignored, so warnings and errors were
        # indistinguishable from ordinary chatter.
        self.log_text.insert(tk.END, formatted_msg, level if level in ("warning", "error") else ())
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        # No update_idletasks() here: log_message runs inside after() callbacks,
        # and pumping the event loop from one invites re-entrant redraws.

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
        try:
            ticker = self.tkr_var.get().strip().upper()
            period = self.period_var.get()
            interval = self.interval_var.get()

            # Served from the local cache when it is fresh. Beyond saving a
            # round trip, this is what makes a backtest reproducible: yfinance
            # revises history, so re-downloading moves the target.
            data, meta = data_cache.get_ohlcv(
                ticker, period, interval,
                adjusted=self.adjusted_var.get(),
                force_refresh=self.force_refresh_var.get(),
            )

            if data.empty:
                raise ValueError(f"No data returned for ticker {ticker}.")

            self.historical_data = data
            self.data_meta = meta
            self.root.after(0, self._on_fetch_success)
        except Exception as e:
            self.root.after(0, self._on_process_error, str(e))

    def _on_fetch_success(self):
        meta = getattr(self, "data_meta", {}) or {}
        origin = meta.get("source", "download")
        detail = f" (from {origin}"
        if origin == "cache":
            detail += f", {meta.get('age_hours', 0)}h old"
        detail += f", {'adjusted' if meta.get('adjusted') else 'raw'} prices)"
        self.log_message(f"Successfully fetched {len(self.historical_data)} data points{detail}.")
        if not meta.get("adjusted", True):
            self.log_message("Raw prices in use: correct for backtesting, but splits will show as jumps.", "warning")
        self.overlay_forecasts.clear()
        # The live forecast belongs to the series it was run against. Keeping it
        # after a refetch left a stale curve on the chart with no way to tell.
        if self.forecast_data is not None:
            self.forecast_data = None
            self.forecast_anchor = None
            self.forecast_target_col = None
            self.forecast_quantiles = None
            self.log_message("Cleared the previous forecast; re-run it against the new data.", "warning")
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

            gaps = int(np.isnan(time_series).sum())
            if gaps:
                # Carry the last observed price across gaps (and back-fill any
                # leading ones). Imputing the series mean, as this used to do,
                # injects prices that never traded anywhere near those dates.
                time_series = pd.Series(time_series).ffill().bfill().to_numpy()
                self.root.after(0, self.log_message,
                                f"Filled {gaps} missing value(s) in '{target_col}' by carrying prices forward.",
                                "warning")
            
            # Move to the modelling space before anything is sliced or padded,
            # so the context length is counted in the units the model sees.
            space = self._selected_space()
            price_series = time_series
            if space != transforms.PRICE:
                time_series = transforms.encode(price_series, space)
                self.root.after(0, self.log_message,
                                f"Modelling {transforms.SPACE_LABELS[space].lower()} "
                                f"({time_series.size} observations after differencing).")

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
                "backend": self.backend_var.get(),
                "mode": EVALUATOR_MODE,
            }
            if EVALUATOR_MODE != "timesfm3":
                # Only the legacy constructor bakes these in. Keying on them in
                # timesfm3 mode forced a full reload whenever the horizon moved,
                # even though ModelConfig never sees them.
                current_request_config["context_len"] = context_len
                current_request_config["horizon"] = horizon
            
            if self.loaded_model is None or self.current_model_config != current_request_config:
                self.root.after(0, self.log_message, f"Hardware loading TimesFM Model from {current_request_config['repo']} (this takes time)...")
                
                device_target = current_request_config["backend"]
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
            
            raw_quantiles = None
            if EVALUATOR_MODE == "timesfm3":
                # TimesFM 3.0 uses predict_batch instead of forecast
                outputs = list(self.loaded_model.predict_batch(
                    [input_context],
                    horizon=horizon,
                    # Was False, which threw away the single most useful thing
                    # the model produces: its uncertainty.
                    return_quantiles=True,
                    use_symmetric_averaging=False
                ))
                forecast_result = outputs[0].forecast
                raw_quantiles = getattr(outputs[0], "quantiles", None)
            else:
                freq_ind = [self.freq_ind_var.get()]
                point_forecast, quantile_forecast = self.loaded_model.forecast(
                    [input_context], freq=freq_ind
                )
                forecast_result = point_forecast[0]
                # Legacy timesfm returns [batch, horizon, 10]: column 0 is the
                # mean, columns 1..9 are deciles. This used to be discarded.
                if quantile_forecast is not None:
                    raw_quantiles = np.asarray(quantile_forecast)[0]

            spread = self._normalise_quantiles(raw_quantiles, horizon)
            if spread:
                self.root.after(0, self.log_message,
                                f"Captured {len(spread['levels'])} quantile levels.")

            # Reconstruct a price path so the plot, the CSV and every saved
            # score stay in price space no matter what was modelled.
            if space != transforms.PRICE:
                anchor_price = float(price_series[-1])
                forecast_result = transforms.decode(
                    np.asarray(forecast_result, dtype=float).reshape(-1), anchor_price, space)
                if spread:
                    matrix = np.asarray(spread["values"], dtype=float)
                    spread = {
                        "levels": spread["levels"],
                        # Each quantile path compounds on its own.
                        "values": np.column_stack([
                            transforms.decode(matrix[:, i], anchor_price, space)
                            for i in range(matrix.shape[1])
                        ]).tolist(),
                    }

            self.forecast_quantiles = spread
            self.forecast_space = space
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
                quantiles=self.forecast_quantiles,
            )
            self.log_message("Forecast results saved to database successfully.")
            self.refresh_forecast_history()
        except Exception as db_e:
            self.log_message(f"Database save failed: {str(db_e)}", "error")

    def _selected_space(self):
        """Map the combobox label back to a transforms constant."""
        label = self.target_space_var.get()
        for space, text in transforms.SPACE_LABELS.items():
            if text == label:
                return space
        return transforms.PRICE

    @staticmethod
    def _normalise_quantiles(raw, horizon):
        """Coerce whatever the backend returned into {levels, values}.

        The two evaluator paths disagree on shape and neither documents it
        firmly, so this is deliberately defensive: anything unrecognised
        degrades to no quantiles rather than a crash mid-forecast.
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
            # column 0 is the mean, not a quantile
            matrix = matrix[:, 1:]
            levels = [round(0.1 * i, 1) for i in range(1, 10)]
        elif width == 9:
            levels = [round(0.1 * i, 1) for i in range(1, 10)]
        else:
            levels = [round((i + 1) / (width + 1), 4) for i in range(width)]

        if not np.isfinite(matrix).all():
            return None
        # Quantiles must not cross; models occasionally emit them unsorted.
        matrix = np.sort(matrix, axis=1)
        return {"levels": levels, "values": matrix.tolist()}

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
        self.log_message(str(error_msg), "error")
        # The dialog used to carry the whole traceback, which pushed the actual
        # message off screen. The log keeps the full text.
        headline = str(error_msg).strip().splitlines()[0] if str(error_msg).strip() else "Unknown error"
        messagebox.showerror("Process Error", headline)
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
            # Raising here escaped into Tk's callback machinery and only ever
            # reached stderr, leaving the user with a silently stale plot.
            self.log_message(
                f"Cannot plot: {len(dates)} dates against {len(prices)} prices.", "error"
            )
            self.canvas.draw()
            return

        self.ax.plot(dates, prices, label="Historical Data", color="blue", linewidth=1.5)
        last_date = dates[-1]
        last_price = float(prices[-1])

        if self.forecast_data is not None:
            forecast_values = np.asarray(self.forecast_data, dtype=float).reshape(-1)
            if forecast_values.size:
                live_anchor = last_date
                live_price = last_price
                if self.forecast_anchor is not None:
                    live_anchor = pd.Timestamp(self.forecast_anchor)
                    if live_anchor.tzinfo is not None:
                        live_anchor = live_anchor.tz_localize(None)
                    live_price = float(pd.Series(prices, index=dates).get(live_anchor, last_price))

                forecast_dates = self._get_forecast_dates(
                    live_anchor, self.interval_var.get(), forecast_values.size
                )
                # Draw the predicted distribution before the point forecast, so
                # the line sits on top of its own uncertainty band.
                self._draw_quantile_band(forecast_dates)

                self.ax.plot(
                    pd.DatetimeIndex([live_anchor]).append(forecast_dates),
                    np.concatenate(([live_price], forecast_values)),
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

    def _draw_quantile_band(self, forecast_dates):
        """Shade the predicted quantile spread as a fan.

        A single line implies a certainty the model never claimed; the band is
        what the forecast actually says.
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
                    matrix[:, levels.index(low)],
                    matrix[:, levels.index(high)],
                    color="orange", alpha=alpha, linewidth=0,
                    label=f"{int((high - low) * 100)}% interval",
                )

    def export_csv(self):
        if self.historical_data is None or self.forecast_data is None:
            messagebox.showinfo("Export", "No complete forecast data to export. Run a forecast first.")
            return
            
        from tkinter import filedialog

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Forecast Data"
        )
        
        if not filepath:
            return
            
        try:
            target = self.forecast_target_col or self.target_col_var.get()
            forecast_values = np.asarray(self.forecast_data, dtype=float).reshape(-1)

            dates = pd.DatetimeIndex(pd.to_datetime(self.historical_data.index))
            if dates.tz is not None:
                dates = dates.tz_localize(None)
            anchor = pd.Timestamp(self.forecast_anchor) if self.forecast_anchor is not None else dates[-1]
            if anchor.tzinfo is not None:
                anchor = anchor.tz_localize(None)
            forecast_dates = self._get_forecast_dates(
                anchor, self.interval_var.get(), forecast_values.size
            )

            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Type", "Date", "Step", target])
                # Dated history alongside the forecast, so the file can be
                # reconciled against actuals later without guessing the anchor.
                for date, value in zip(dates, np.asarray(self.historical_data[target], dtype=float).reshape(-1)):
                    writer.writerow(["Historical", date.isoformat(), "", value])
                for step, (date, value) in enumerate(zip(forecast_dates, forecast_values), start=1):
                    writer.writerow(["Forecast", date.isoformat(), step, value])

            self.log_message(f"Forecast successfully exported to {filepath}")
        except Exception as e:
            self.log_message(f"Export Error: {str(e)}", "error")

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
            mase = record.get('mase_score')
            direction = record.get('directional_accuracy')
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
                "-" if mae is None else f"{mae:.4f}",
                "-" if mase is None else f"{mase:.3f}",
                "-" if direction is None else f"{direction * 100:.0f}%"
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

            y_true = aligned.values[mask]
            y_pred = forecast_values[mask]

            mae = metrics.mae(y_true, y_pred)

            # MASE denominator comes from the context that preceded the anchor,
            # so the scale never sees data the forecast could not have seen.
            context = actuals[actuals.index <= anchor].to_numpy(float)
            mase = metrics.mase(y_true, y_pred, context, seasonality=1)

            anchor_value = float(actuals.get(anchor, np.nan))
            direction = metrics.directional_accuracy(y_true, y_pred, anchor_value)

            # What the naive forecast would have scored on the same points.
            naive_mae = metrics.mae(y_true, np.full(y_true.size, anchor_value))

            try:
                db_manager.update_scores(record["id"], mae_score=mae,
                                         mase_score=None if np.isnan(mase) else mase,
                                         directional_accuracy=None if np.isnan(direction) else direction)
            except Exception as e:
                self.log_message(f"Could not save scores for #{record['id']}: {str(e)}", "error")
                continue

            scored += 1
            verdict = "beats naive" if np.isfinite(mae) and np.isfinite(naive_mae) and mae < naive_mae else "does NOT beat naive"
            self.log_message(
                f"Forecast #{record['id']} ({target_column}): MAE {mae:.4f} vs naive "
                f"{naive_mae:.4f}, MASE {mase:.3f}, direction {direction * 100:.0f}% "
                f"over {int(mask.sum())}/{forecast_values.size} point(s) - {verdict}.",
                "info" if mae < naive_mae else "warning"
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