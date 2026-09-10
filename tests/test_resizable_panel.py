"""The settings column is draggable, and the widgets inside follow it."""

import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    _root = tk.Tk(); _root.geometry("1400x850")
except Exception as exc:
    print(f"SKIPPED: Tk unavailable ({exc})"); raise SystemExit(0)

import db_manager as db
db.DB_DIR = tempfile.mkdtemp(); db.DB_PATH = os.path.join(db.DB_DIR, 'pane.db')
import timesfm_forecaster as forecaster

app = forecaster.TimesFMApp(_root)
_root.update_idletasks(); _root.update()
_root.update_idletasks()          # let the after_idle sash placement run

paned = app.inference_paned
assert str(paned.cget("orient")) == "horizontal"
panes = paned.panes()
assert len(panes) == 2, panes
print(f"paned window with {len(panes)} panes, width {paned.winfo_width()}px")

start = paned.sashpos(0)
print(f"  divider starts at {start}px")
assert 200 < start < 900, start

# ---- the divider actually moves -------------------------------------------
paned.sashpos(0, 620)
_root.update_idletasks(); _root.update()
widened = paned.sashpos(0)
print(f"  dragged to 620 -> {widened}px")
assert widened > start + 100, (start, widened)

left = _root.nametowidget(panes[0])
right = _root.nametowidget(panes[1])
print(f"  left pane {left.winfo_width()}px, right pane {right.winfo_width()}px")
assert left.winfo_width() > 500, left.winfo_width()

# ---- the settings canvas follows the pane, not a frozen 340px --------------
canvas = app.settings_frame.nametowidget(app.settings_frame.winfo_parent())
print(f"  settings canvas now {canvas.winfo_width()}px wide")
assert canvas.winfo_width() > 450, canvas.winfo_width()

# content stretches to fill, so a widened panel is not dead space
item = canvas.find_all()[0]
content_width = int(canvas.itemcget(item, "width"))
print(f"  content window stretched to {content_width}px")
assert content_width >= canvas.winfo_width() - 4, (content_width, canvas.winfo_width())

# ---- narrowing keeps the horizontal scrollbar meaningful -------------------
paned.sashpos(0, 240)
_root.update_idletasks(); _root.update()
narrow = paned.sashpos(0)
natural = app.settings_frame.winfo_reqwidth()
item_width = int(canvas.itemcget(canvas.find_all()[0], "width"))
print(f"  narrowed to {narrow}px; content clamped at {item_width}px (natural {natural}px)")
assert item_width >= natural - 4, (item_width, natural)
region = [float(v) for v in canvas.cget("scrollregion").split()]
assert region[2] > canvas.winfo_width(), (region[2], canvas.winfo_width())
print("  content stays at its natural width, so horizontal scrolling still works")

# ---- the plot keeps the slack when the window grows -----------------------
before_right = right.winfo_width()
_root.geometry("1700x850"); _root.update_idletasks(); _root.update()
after_right = right.winfo_width()
print(f"  window +300px: right pane {before_right} -> {after_right}px")
assert after_right > before_right, "extra width did not go to the plot"

_root.destroy()
print("\nALL RESIZABLE-PANEL TESTS PASSED")
