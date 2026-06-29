"""
BLS Meta-Analysis Pipeline -- Standalone Launcher
==================================================
Entry point for the packaged .exe.
Double-click -> UI opens -> configure -> Apply & Run -> results saved.

Looks for config.yaml next to the exe (or this script).
Creates a default config.yaml on first run.
"""
from __future__ import annotations
import sys, os, threading, traceback, io, contextlib

# -- Path setup (PyInstaller frozen vs. dev) -----------------------------------
if getattr(sys, "frozen", False):
    _BASE = os.path.dirname(sys.executable)   # directory next to the .exe
    _CODE = sys._MEIPASS                       # bundled code temp dir
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))
    _CODE = _BASE

sys.path.insert(0, _CODE)

_CFG_PATH = os.path.join(_BASE, "config.yaml")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import webbrowser
import yaml  # type: ignore

# -- Default config if none exists ---------------------------------------------
_DEFAULT_CFG = {
    "data": {"input_path": "", "output_dir": ""},
    "analysis": {
        "alpha": 0.10, "min_n": 5,
        "weight_col": "iv_weight",
        "question_tags": ["AD_RECALL", "AWARENESS", "FAVORABILITY", "INTENT"],
        "mt_correction": "none",
    },
    "preprocessing": {"winsorize_lower": 0.02, "winsorize_upper": 0.98, "freq_p99_cap": True},
    "cuts": {c: {"enabled": True, "min_n": 5} for c in [
        "Watch Time", "Creative Count", "Video Duration", "VCR",
        "Frequency", "Weekly Impressions", "Objective", "Product Split",
        "Account Segment", "Audience Type", "Billing Type", "Spark Ads", "ACO",
    ]},
    "cross_cuts": {"enabled": True, "min_n_multiplier": 3,
                   "pairs": [["Watch Time", "VCR"], ["Watch Time", "Objective"]]},
    "feature_importance": {
        "enabled": True, "n_estimators": 300, "cv_folds": 5,
        "top_n": 15, "stratify_by_tag": False,
    },
    "output": {"save_figures": True, "export_csv": True, "export_excel": False, "dpi": 150},
    "custom_cuts": [],
    "report": {
        "enabled": True,
        "ai_api_key": "",
        "ai_model": "gemini-2.0-flash",
    },
}

def _load_cfg() -> dict:
    if os.path.exists(_CFG_PATH):
        with open(_CFG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return _DEFAULT_CFG.copy()

def _save_cfg(cfg: dict) -> None:
    with open(_CFG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ==============================================================================
#  CONSTANTS
# ==============================================================================
ALL_CUTS = [
    "Watch Time", "Creative Count", "Video Duration", "VCR",
    "Frequency", "Weekly Impressions", "Objective", "Product Split",
    "Account Segment", "Audience Type", "Billing Type", "Spark Ads", "ACO",
]
ALL_QUESTION_TAGS = ["AD_RECALL", "AWARENESS", "FAVORABILITY", "INTENT"]
ALPHA_OPTIONS     = [0.01, 0.05, 0.10]
WEIGHT_OPTIONS    = ["iv_weight", "n_weight"]
DPI_OPTIONS       = [100, 150, 200, 300]
CV_FOLD_OPTIONS       = [3, 5, 10]
MT_CORRECTION_OPTIONS = ["none", "bh", "bonferroni", "holm"]
TIER1_PAIRS = [
    ("Watch Time",     "VCR"),
    ("Watch Time",     "Objective"),
    ("Video Duration", "VCR"),
    ("Creative Count", "Objective"),
    ("Frequency",      "Audience Type"),
]
TIER2_PAIRS = [
    ("Watch Time",     "Creative Count"),
    ("Billing Type",   "Objective"),
    ("Spark Ads",      "Audience Type"),
    ("ACO",            "Creative Count"),
    ("VCR",            "Frequency"),
]


# ==============================================================================
#  MAIN APP
# ==============================================================================
def build_app(root: tk.Tk) -> None:
    cfg = _load_cfg()
    root.title("BLS Meta-Analysis Pipeline")
    root.geometry("920x740")
    root.resizable(True, True)
    try: root.tk.call("tk", "scaling", 1.15)
    except Exception: pass
    style = ttk.Style()
    try: style.theme_use("clam")
    except Exception: pass

    # -- Scrollable frame helper -----------------------------------------------
    def scrollable(parent) -> ttk.Frame:
        cont   = ttk.Frame(parent)
        cont.pack(fill="both", expand=True)
        canvas = tk.Canvas(cont, highlightthickness=0)
        vbar   = ttk.Scrollbar(cont, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        def _scroll(event):
            if event.num == 4:   canvas.yview_scroll(-1, "units")
            elif event.num == 5: canvas.yview_scroll(1,  "units")
            else:                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _scroll)
        canvas.bind_all("<Button-4>",   _scroll)
        canvas.bind_all("<Button-5>",   _scroll)
        return inner

    def heading(parent, text, row):
        ttk.Label(parent, text=text, font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=4, sticky="w", padx=8, pady=(10, 2))

    def hint(parent, text, row, col=3):
        ttk.Label(parent, text=text, foreground="#777").grid(
            row=row, column=col, sticky="w", padx=4)

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    # --- TAB 0: Data Paths ----------------------------------------------------
    tab0 = ttk.Frame(nb);  nb.add(tab0, text="[1] Data Paths")
    f0   = scrollable(tab0)
    heading(f0, "Data Paths", 0)
    ttk.Label(f0, text="Set input CSV and output folder, then click Apply & Run.",
              foreground="#666").grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))

    data_cfg = cfg.get("data", {})
    v_input  = tk.StringVar(value=data_cfg.get("input_path", ""))
    v_output = tk.StringVar(value=data_cfg.get("output_dir", ""))

    r = 2
    ttk.Label(f0, text="Input CSV:").grid(row=r, column=0, sticky="w", padx=8, pady=4)
    ttk.Entry(f0, textvariable=v_input, width=55).grid(row=r, column=1, columnspan=2, sticky="ew", padx=4)
    def browse_csv():
        p = filedialog.askopenfilename(title="Select input CSV",
                                       filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if p: v_input.set(p)
    ttk.Button(f0, text="Browse...", command=browse_csv).grid(row=r, column=3, padx=4)
    r += 1

    ttk.Label(f0, text="Output folder:").grid(row=r, column=0, sticky="w", padx=8, pady=4)
    ttk.Entry(f0, textvariable=v_output, width=55).grid(row=r, column=1, columnspan=2, sticky="ew", padx=4)
    def browse_out():
        p = filedialog.askdirectory(title="Select output folder")
        if p: v_output.set(p)
    ttk.Button(f0, text="Browse...", command=browse_out).grid(row=r, column=3, padx=4)
    r += 1

    ttk.Separator(f0, orient="horizontal").grid(row=r, columnspan=4, sticky="ew", padx=8, pady=10); r += 1
    ttk.Label(f0, text="Output folder will be created if it does not exist.",
              foreground="#888").grid(row=r, column=0, columnspan=4, sticky="w", padx=8); r += 1
    ttk.Label(f0, text="Configure analysis settings in tabs [2]-[5], then click Apply & Run.",
              foreground="#555").grid(row=r, column=0, columnspan=4, sticky="w", padx=8); r += 1

    # --- TAB 1: Analysis + Preprocessing --------------------------------------
    tab1 = ttk.Frame(nb);  nb.add(tab1, text="[2] Analysis")
    f1   = scrollable(tab1)
    ana  = cfg.get("analysis", {})
    pre  = cfg.get("preprocessing", {})
    heading(f1, "Analysis Settings", 0)

    r = 1
    ttk.Label(f1, text="Alpha:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_alpha = tk.StringVar(value=str(ana.get("alpha", 0.10)))
    ttk.Combobox(f1, textvariable=v_alpha, values=[str(v) for v in ALPHA_OPTIONS],
                 width=8, state="readonly").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "0.01 strict  |  0.05 standard  |  0.10 lenient", r); r += 1

    ttk.Label(f1, text="MT Correction:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_mt_corr = tk.StringVar(value=ana.get("mt_correction", "none"))
    ttk.Combobox(f1, textvariable=v_mt_corr, values=MT_CORRECTION_OPTIONS,
                 width=12, state="readonly").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "BH = FDR (recommended)  |  bonferroni = conservative  |  holm = stepwise", r); r += 1

    ttk.Label(f1, text="Weight col:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_weight = tk.StringVar(value=ana.get("weight_col", "iv_weight"))
    ttk.Combobox(f1, textvariable=v_weight, values=WEIGHT_OPTIONS,
                 width=14, state="readonly").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "iv_weight = 1/SE^2  |  n_weight = harmonic N", r); r += 1

    ttk.Label(f1, text="Global min_n:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_min_n = tk.IntVar(value=ana.get("min_n", 5))
    ttk.Spinbox(f1, from_=2, to=50, textvariable=v_min_n,
                width=6).grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "min campaigns per group bin  (range 2-50)", r); r += 1

    ttk.Label(f1, text="Question Tags:", font=("", 9, "bold")).grid(
        row=r, column=0, sticky="nw", padx=8, pady=3)
    active_tags = ana.get("question_tags", ALL_QUESTION_TAGS)
    qt_vars = {}
    for qt in ALL_QUESTION_TAGS:
        v = tk.BooleanVar(value=(qt in active_tags))
        ttk.Checkbutton(f1, text=qt, variable=v).grid(row=r, column=1, sticky="w", padx=4, pady=1)
        qt_vars[qt] = v; r += 1

    ttk.Separator(f1, orient="horizontal").grid(row=r, columnspan=4, sticky="ew", padx=8, pady=6); r += 1
    heading(f1, "Preprocessing", r); r += 1

    ttk.Label(f1, text="Winsorize lower:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_wlo = tk.DoubleVar(value=pre.get("winsorize_lower", 0.02))
    ttk.Spinbox(f1, from_=0.01, to=0.05, increment=0.01, textvariable=v_wlo,
                width=7, format="%.2f").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "lower lift clip percentile  (0.01-0.05)", r); r += 1

    ttk.Label(f1, text="Winsorize upper:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_whi = tk.DoubleVar(value=pre.get("winsorize_upper", 0.98))
    ttk.Spinbox(f1, from_=0.95, to=0.99, increment=0.01, textvariable=v_whi,
                width=7, format="%.2f").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "upper lift clip percentile  (0.95-0.99)", r); r += 1

    v_freq_cap = tk.BooleanVar(value=pre.get("freq_p99_cap", True))
    ttk.Checkbutton(f1, text="Cap frequency at P99 before binning",
                    variable=v_freq_cap).grid(row=r, column=0, columnspan=3, sticky="w", padx=8, pady=3)

    # --- TAB 2: Cuts ----------------------------------------------------------
    tab2 = ttk.Frame(nb);  nb.add(tab2, text="[3] Cuts")
    f2   = scrollable(tab2)
    cuts_cfg = cfg.get("cuts", {})
    ttk.Label(f2, text="Cut Settings", font=("", 10, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 2))
    for txt, col in [("Cut Name", 0), ("Enabled", 1), ("min_n", 2)]:
        ttk.Label(f2, text=txt, font=("", 9, "bold")).grid(row=1, column=col, padx=10)
    cut_en_vars = {}; cut_minn_vars = {}
    for idx, cut in enumerate(ALL_CUTS):
        cc_c = cuts_cfg.get(cut, {})
        r2   = idx + 2
        ttk.Label(f2, text=cut).grid(row=r2, column=0, sticky="w", padx=8, pady=2)
        v_en = tk.BooleanVar(value=cc_c.get("enabled", True))
        v_mn = tk.IntVar(value=cc_c.get("min_n", ana.get("min_n", 5)))
        ttk.Checkbutton(f2, variable=v_en).grid(row=r2, column=1, padx=10)
        ttk.Spinbox(f2, from_=2, to=50, textvariable=v_mn, width=6).grid(row=r2, column=2, padx=10)
        cut_en_vars[cut] = v_en; cut_minn_vars[cut] = v_mn

    # -- Custom cut state shared by Tab 3 & Tab 6 ----------------------------
    custom_cuts_data = list(cfg.get("custom_cuts", []))

    # --- TAB 3: Cross-Cuts ----------------------------------------------------
    tab3 = ttk.Frame(nb);  nb.add(tab3, text="[4] Cross-Cuts")
    f3   = scrollable(tab3)
    cc   = cfg.get("cross_cuts", {})
    ttk.Label(f3, text="Cross-Cut Analysis", font=("", 10, "bold")).grid(
        row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 2))
    v_cc_en   = tk.BooleanVar(value=cc.get("enabled", True))
    v_cc_mult = tk.IntVar(value=cc.get("min_n_multiplier", 3))
    ttk.Checkbutton(f3, text="Enable cross-cut analysis",
                    variable=v_cc_en).grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=3)
    ttk.Label(f3, text="min_n multiplier:").grid(row=2, column=0, sticky="w", padx=8, pady=3)
    ttk.Spinbox(f3, from_=2, to=10, textvariable=v_cc_mult, width=6).grid(
        row=2, column=1, sticky="w", padx=4)
    hint(f3, "cross min_n = global_min_n x this  (range 2-10)", 2)
    ttk.Label(f3, text="Select cross-cut pairs:", font=("", 9, "bold")).grid(
        row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 2))
    ttk.Label(f3, text="(check pairs to analyze as 2-D heatmaps)",
              foreground="#777").grid(row=4, column=0, columnspan=4, sticky="w", padx=8)

    _tier1_strs = {f"{p[0]} x {p[1]}" for p in TIER1_PAIRS}
    _saved_pairs = cc.get("pairs")  # None = key absent, [] = user cleared all
    if _saved_pairs is None:
        # First run / no saved config: default to tier1
        existing_pairs_set = _tier1_strs
    else:
        # Respect saved config exactly, even if empty
        existing_pairs_set = {f"{p[0]} x {p[1]}" for p in _saved_pairs}
    cc_pair_vars = {}
    r3 = 5

    ttk.Label(f3, text="⭐ Tier 1 — Recommended (default on):",
              font=("", 9, "bold")).grid(
        row=r3, column=0, columnspan=4, sticky="w", padx=8, pady=(6, 2)); r3 += 1
    for pair in TIER1_PAIRS:
        pair_str = f"{pair[0]} x {pair[1]}"
        v_pair   = tk.BooleanVar(value=(pair_str in existing_pairs_set))
        ttk.Checkbutton(f3, text=pair_str, variable=v_pair).grid(
            row=r3, column=0, columnspan=4, sticky="w", padx=20, pady=1)
        cc_pair_vars[pair_str] = v_pair; r3 += 1

    ttk.Label(f3, text="◎ Tier 2 — Optional (default off):",
              font=("", 9, "bold")).grid(
        row=r3, column=0, columnspan=4, sticky="w", padx=8, pady=(10, 2)); r3 += 1
    for pair in TIER2_PAIRS:
        pair_str = f"{pair[0]} x {pair[1]}"
        v_pair   = tk.BooleanVar(value=(pair_str in existing_pairs_set))
        ttk.Checkbutton(f3, text=pair_str, variable=v_pair).grid(
            row=r3, column=0, columnspan=4, sticky="w", padx=20, pady=1)
        cc_pair_vars[pair_str] = v_pair; r3 += 1

    # Container for custom-cut cross pairs (rebuilt by Tab 6 callbacks)
    custom_cc_container = ttk.Frame(f3)
    custom_cc_container.grid(row=r3, column=0, columnspan=4, sticky="ew")

    # --- TAB 4: Feature Importance + Output -----------------------------------
    tab4 = ttk.Frame(nb);  nb.add(tab4, text="[5] FI & Output")
    f4   = scrollable(tab4)
    fi   = cfg.get("feature_importance", {})
    out  = cfg.get("output", {})
    heading(f4, "Feature Importance (Random Forest)", 0)
    v_fi_en = tk.BooleanVar(value=fi.get("enabled", True))
    ttk.Checkbutton(f4, text="Enable feature importance",
                    variable=v_fi_en).grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=3)
    v_fi_strat = tk.BooleanVar(value=fi.get("stratify_by_tag", False))
    ttk.Checkbutton(
        f4,
        text="Stratify FI by question tag  (separate RF per tag, side-by-side comparison)",
        variable=v_fi_strat,
    ).grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=2)
    r4 = 3
    ttk.Label(f4, text="RF trees:").grid(row=r4, column=0, sticky="w", padx=8, pady=3)
    v_fi_nest = tk.IntVar(value=fi.get("n_estimators", 300))
    ttk.Spinbox(f4, from_=100, to=500, increment=50, textvariable=v_fi_nest,
                width=8).grid(row=r4, column=1, sticky="w", padx=4)
    hint(f4, "100-500  (more = stable, slower)", r4); r4 += 1
    ttk.Label(f4, text="CV folds:").grid(row=r4, column=0, sticky="w", padx=8, pady=3)
    v_fi_cv = tk.StringVar(value=str(fi.get("cv_folds", 5)))
    ttk.Combobox(f4, textvariable=v_fi_cv, values=[str(v) for v in CV_FOLD_OPTIONS],
                 width=6, state="readonly").grid(row=r4, column=1, sticky="w", padx=4)
    hint(f4, "options: 3 | 5 | 10", r4); r4 += 1
    ttk.Label(f4, text="Top N features:").grid(row=r4, column=0, sticky="w", padx=8, pady=3)
    v_fi_topn = tk.IntVar(value=fi.get("top_n", 15))
    ttk.Spinbox(f4, from_=5, to=30, textvariable=v_fi_topn,
                width=6).grid(row=r4, column=1, sticky="w", padx=4)
    hint(f4, "5-30", r4); r4 += 1
    ttk.Separator(f4, orient="horizontal").grid(row=r4, columnspan=4, sticky="ew", padx=8, pady=6); r4 += 1
    heading(f4, "Output Settings", r4); r4 += 1
    ttk.Label(f4, text="Figure DPI:").grid(row=r4, column=0, sticky="w", padx=8, pady=3)
    v_dpi = tk.StringVar(value=str(out.get("dpi", 150)))
    ttk.Combobox(f4, textvariable=v_dpi, values=[str(v) for v in DPI_OPTIONS],
                 width=6, state="readonly").grid(row=r4, column=1, sticky="w", padx=4)
    hint(f4, "100 draft  |  150 default  |  300 print-quality", r4); r4 += 1
    v_save_figs  = tk.BooleanVar(value=out.get("save_figures", True))
    v_export_csv = tk.BooleanVar(value=out.get("export_csv",   True))
    v_export_xl  = tk.BooleanVar(value=out.get("export_excel", False))
    for txt, var in [("Save PNG figures", v_save_figs),
                     ("Export CSV",       v_export_csv),
                     ("Export Excel (.xlsx)", v_export_xl)]:
        ttk.Checkbutton(f4, text=txt, variable=var).grid(
            row=r4, column=0, columnspan=3, sticky="w", padx=8, pady=2); r4 += 1

    ttk.Separator(f4, orient="horizontal").grid(
        row=r4, columnspan=4, sticky="ew", padx=8, pady=6); r4 += 1
    heading(f4, "HTML Report", r4); r4 += 1

    rpt_cfg = cfg.get("report", {})
    v_rpt_en = tk.BooleanVar(value=rpt_cfg.get("enabled", True))
    ttk.Checkbutton(f4, text="Generate HTML report after run",
                    variable=v_rpt_en).grid(
        row=r4, column=0, columnspan=3, sticky="w", padx=8, pady=3); r4 += 1

    # ── AI provider & model auto-detect ───────────────────────────────────────
    # Key prefix → (provider_label, default_model)
    _KEY_PROFILES = {
        "AIza":        ("Gemini",    "gemini-2.0-flash"),
        "sk-ant":      ("Anthropic", "claude-haiku-4-5"),
        "gsk_":        ("Groq",      "llama-3.3-70b-versatile"),
        "sk-proj-":    ("OpenAI",    "gpt-4o-mini"),
        "sk-o1-":      ("OpenAI",    "o1-mini"),
        "sk-":         ("OpenAI",    "gpt-4o-mini"),
    }
    _KEY_HINT_LINKS = {
        "Gemini":    "aistudio.google.com (free)",
        "Groq":      "console.groq.com (free)",
        "Anthropic": "console.anthropic.com (paid)",
        "OpenAI":    "platform.openai.com (paid)",
    }
    _MODEL_OPTIONS = [
        "gemini-2.0-flash",         # Gemini free (recommended)
        "gemini-1.5-flash",         # Gemini free
        "llama-3.3-70b-versatile",  # Groq free (recommended)
        "llama3-8b-8192",           # Groq free (faster)
        "gemma2-9b-it",             # Groq free
        "gpt-4o-mini",              # OpenAI cheap
        "gpt-4o",                   # OpenAI
        "o1-mini",                  # OpenAI
        "claude-haiku-4-5",         # Anthropic cheapest (current)
        "claude-sonnet-4-6",        # Anthropic mid (current)
        "claude-opus-4-8",          # Anthropic powerful (current)
    ]

    def _detect_provider_from_key(k: str):
        """Return (provider_label, default_model) or None."""
        for prefix, profile in _KEY_PROFILES.items():
            if k.startswith(prefix):
                return profile
        return None

    # Saved values
    _saved_key = rpt_cfg.get("ai_api_key", "")
    _saved_model = rpt_cfg.get("ai_model", "gemini-2.0-flash")
    v_key_mode = tk.StringVar(value="builtin" if not _saved_key else "custom")
    v_ai_key   = tk.StringVar(value=_saved_key)
    v_ai_model = tk.StringVar(value=_saved_model)

    # API Key row
    ttk.Label(f4, text="API Key:").grid(row=r4, column=0, sticky="w", padx=8, pady=3)
    rb_builtin = ttk.Radiobutton(f4, text="Built-in (Gemini Flash, free)",
                                  variable=v_key_mode, value="builtin")
    rb_custom  = ttk.Radiobutton(f4, text="Own key:",
                                  variable=v_key_mode, value="custom")
    rb_builtin.grid(row=r4, column=1, sticky="w", padx=4); r4 += 1
    rb_custom.grid(row=r4, column=1, sticky="w", padx=4)
    ai_key_entry = ttk.Entry(f4, textvariable=v_ai_key, width=36, show="*")
    ai_key_entry.grid(row=r4, column=2, columnspan=2, sticky="ew", padx=4)
    def _toggle_key():
        ai_key_entry.config(show="" if ai_key_entry.cget("show") == "*" else "*")
    ttk.Button(f4, text="👁", width=3, command=_toggle_key).grid(row=r4, column=3, padx=2); r4 += 1

    # Auto-detect hint label (green = detected, orange = warning, blank = ok)
    _key_hint_var = tk.StringVar(value="")
    _key_hint_lbl = ttk.Label(f4, textvariable=_key_hint_var, font=("TkDefaultFont", 8))
    _key_hint_lbl.grid(row=r4, column=1, columnspan=3, sticky="w", padx=4); r4 += 1

    # AI Model row (editable combobox — auto-filled, user can override)
    ttk.Label(f4, text="AI Model:").grid(row=r4, column=0, sticky="w", padx=8, pady=3)
    _model_cb = ttk.Combobox(f4, textvariable=v_ai_model, values=_MODEL_OPTIONS,
                              width=32, state="normal")
    _model_cb.grid(row=r4, column=1, columnspan=2, sticky="w", padx=4)
    ttk.Label(f4, text="(auto-filled from key; editable)", foreground="#999",
              font=("TkDefaultFont", 8)).grid(row=r4, column=3, sticky="w", padx=4); r4 += 1

    def _update_key_hint(*_):
        """Auto-detect provider from key prefix, update model + hint."""
        if v_key_mode.get() != "custom":
            _key_hint_var.set("")
            _key_hint_lbl.config(foreground="#777")
            return
        k = v_ai_key.get().strip()
        if not k:
            _key_hint_var.set("")
            return
        profile = _detect_provider_from_key(k)
        if profile:
            provider, default_model = profile
            # Auto-set model only if it doesn't match this provider yet
            current = v_ai_model.get()
            provider_keywords = {
                "Gemini": "gemini", "Anthropic": "claude",
                "Groq": ("llama","mixtral","gemma"), "OpenAI": ("gpt","o1","o3"),
            }
            kws = provider_keywords.get(provider, ())
            kws = (kws,) if isinstance(kws, str) else kws
            if not any(kw in current.lower() for kw in kws):
                v_ai_model.set(default_model)
            link = _KEY_HINT_LINKS.get(provider, "")
            _key_hint_var.set(f"✓ {provider} key  →  {v_ai_model.get()}  |  {link}")
            _key_hint_lbl.config(foreground="#16a34a")
        else:
            # Unknown prefix — treat as OpenAI-compatible, don't change model
            _key_hint_var.set("⚠ Unknown key format — routing to OpenAI-compatible endpoint")
            _key_hint_lbl.config(foreground="#e65100")

    v_ai_key.trace_add("write", _update_key_hint)
    v_key_mode.trace_add("write", _update_key_hint)
    # Trigger once on load to show status of saved key
    _update_key_hint()

    ttk.Label(
        f4,
        text="Free keys: Gemini → aistudio.google.com   Groq → console.groq.com",
        foreground="#777",
    ).grid(row=r4, column=0, columnspan=4, sticky="w", padx=8); r4 += 1

    # --- TAB 6: Custom Cuts --------------------------------------------------
    tab6 = ttk.Frame(nb);  nb.add(tab6, text="[6] Custom Cuts")
    f6   = scrollable(tab6)
    heading(f6, "Custom Cuts", 0)
    ttk.Label(
        f6,
        text="Define additional CSV columns as analysis cuts.  "
             "New cuts appear in the Cross-Cuts tab automatically.",
        foreground="#555",
    ).grid(row=1, column=0, columnspan=5, sticky="w", padx=8, pady=(0, 6))

    # Treeview table ------------------------------------------------------------
    tree_outer = ttk.Frame(f6)
    tree_outer.grid(row=2, column=0, columnspan=5, sticky="nsew", padx=8, pady=4)
    tree = ttk.Treeview(
        tree_outer,
        columns=("Name", "Column", "Type", "DF", "min_n", "On"),
        show="headings", height=9,
    )
    for hdr, w_px in [("Name",160),("Column",140),("Type",105),("DF",75),("min_n",55),("On",45)]:
        tree.heading(hdr, text=hdr)
        tree.column(hdr, width=w_px, anchor="w")
    tree_sb = ttk.Scrollbar(tree_outer, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_sb.set)
    tree.pack(side="left", fill="both", expand=True)
    tree_sb.pack(side="right", fill="y")

    def _refresh_tree():
        tree.delete(*tree.get_children())
        for cc in custom_cuts_data:
            tree.insert("", "end", values=(
                cc.get("name",""), cc.get("col",""),
                cc.get("bin_type","categorical"),
                cc.get("df_key","main"),
                cc.get("min_n", 5),
                "Yes" if cc.get("enabled", True) else "No"))

    _refresh_tree()

    # Rebuild custom cross-cut section in Tab 3 --------------------------------
    def _rebuild_custom_cc():
        for child in custom_cc_container.winfo_children():
            child.destroy()
        custom_names = {c["name"] for c in custom_cuts_data if c.get("name")}
        for k in [k for k in list(cc_pair_vars.keys())
                  if k.split(" x ")[0] in custom_names]:
            del cc_pair_vars[k]
        if not custom_cuts_data:
            return
        saved_pairs = {f"{p[0]} x {p[1]}"
                       for p in cfg.get("cross_cuts", {}).get("pairs", [])}
        row_cc = 0
        ttk.Label(custom_cc_container,
                  text="🔧 Custom × Standard Cuts:",
                  font=("", 9, "bold")).grid(
            row=row_cc, column=0, columnspan=4,
            sticky="w", padx=8, pady=(10, 2))
        row_cc += 1
        for ccut in custom_cuts_data:
            cn = ccut.get("name", "").strip()
            if not cn:
                continue
            for std_cut in ALL_CUTS:
                pair_str = f"{cn} x {std_cut}"
                v_pair = tk.BooleanVar(value=(pair_str in saved_pairs))
                ttk.Checkbutton(custom_cc_container, text=pair_str,
                                variable=v_pair).grid(
                    row=row_cc, column=0, columnspan=4,
                    sticky="w", padx=20, pady=1)
                cc_pair_vars[pair_str] = v_pair
                row_cc += 1

    _rebuild_custom_cc()   # initial populate from pre-existing config

    # Add Cut dialog -----------------------------------------------------------
    def _add_cut_dialog():
        dlg = tk.Toplevel(root)
        dlg.title("Add Custom Cut")
        dlg.geometry("510x460")
        dlg.resizable(False, False)
        dlg.grab_set()
        r_d = 0
        def _lbl(t): return ttk.Label(dlg, text=t)
        def _fhint(t): return ttk.Label(dlg, text=t, foreground="#777")

        _lbl("Cut name:").grid(row=r_d, column=0, sticky="w", padx=12, pady=4)
        v_name = tk.StringVar()
        ttk.Entry(dlg, textvariable=v_name, width=28).grid(
            row=r_d, column=1, columnspan=2, sticky="ew", padx=4); r_d += 1

        _lbl("CSV column:").grid(row=r_d, column=0, sticky="w", padx=12, pady=4)
        v_col = tk.StringVar()
        ttk.Entry(dlg, textvariable=v_col, width=28).grid(
            row=r_d, column=1, columnspan=2, sticky="ew", padx=4); r_d += 1

        _lbl("DataFrame:").grid(row=r_d, column=0, sticky="w", padx=12, pady=4)
        v_dfkey = tk.StringVar(value="main")
        ttk.Combobox(dlg, textvariable=v_dfkey,
                     values=["main","product","objective"],
                     width=13, state="readonly").grid(
            row=r_d, column=1, sticky="w", padx=4)
        _fhint("main = campaign rows").grid(
            row=r_d, column=2, sticky="w", padx=4); r_d += 1

        _lbl("Bin type:").grid(row=r_d, column=0, sticky="w", padx=12, pady=4)
        v_bintype = tk.StringVar(value="categorical")
        ttk.Combobox(dlg, textvariable=v_bintype,
                     values=["categorical","quantile","fixed"],
                     width=13, state="readonly").grid(
            row=r_d, column=1, sticky="w", padx=4)
        _fhint("categorical = values as-is").grid(
            row=r_d, column=2, sticky="w", padx=4); r_d += 1

        opts = ttk.LabelFrame(dlg, text="Bin Options (quantile / fixed only)")
        opts.grid(row=r_d, column=0, columnspan=3, sticky="ew", padx=12, pady=6); r_d += 1
        ttk.Label(opts, text="Q (quantiles):").grid(row=0,column=0,sticky="w",padx=8,pady=3)
        v_q = tk.IntVar(value=3)
        ttk.Spinbox(opts,from_=2,to=10,textvariable=v_q,width=5).grid(row=0,column=1,sticky="w",padx=4)
        ttk.Label(opts,text="for quantile mode",foreground="#777").grid(row=0,column=2,sticky="w",padx=4)
        ttk.Label(opts,text="Labels (comma-sep):").grid(row=1,column=0,sticky="w",padx=8,pady=3)
        v_labels = tk.StringVar(value="Low,Mid,High")
        ttk.Entry(opts,textvariable=v_labels,width=28).grid(row=1,column=1,columnspan=2,sticky="ew",padx=4)
        ttk.Label(opts,text='e.g. "Low,Mid,High"  (leave blank = auto)',foreground="#777").grid(row=2,column=0,columnspan=3,sticky="w",padx=8)
        ttk.Label(opts,text="Bins (comma-sep):").grid(row=3,column=0,sticky="w",padx=8,pady=(8,3))
        v_bins_str = tk.StringVar(value="0,100,500,9999")
        ttk.Entry(opts,textvariable=v_bins_str,width=28).grid(row=3,column=1,columnspan=2,sticky="ew",padx=4)
        ttk.Label(opts,text='e.g. "0,100,500,9999"  numeric edges (fixed mode)',foreground="#777").grid(row=4,column=0,columnspan=3,sticky="w",padx=8)

        _lbl("min_n:").grid(row=r_d,column=0,sticky="w",padx=12,pady=4)
        v_min = tk.IntVar(value=5)
        ttk.Spinbox(dlg,from_=2,to=50,textvariable=v_min,width=6).grid(row=r_d,column=1,sticky="w",padx=4)
        _fhint("min campaigns per group").grid(row=r_d,column=2,sticky="w",padx=4); r_d += 1

        err_var = tk.StringVar()
        ttk.Label(dlg,textvariable=err_var,foreground="red").grid(
            row=r_d,column=0,columnspan=3,sticky="w",padx=12); r_d += 1

        def _confirm():
            name     = v_name.get().strip()
            col_name = v_col.get().strip()
            bin_type = v_bintype.get()
            if not name:
                err_var.set("Cut name is required."); return
            if not col_name:
                err_var.set("CSV column name is required."); return
            if name in [c["name"] for c in custom_cuts_data]:
                err_var.set(f"'{name}' already exists."); return
            if name in ALL_CUTS:
                err_var.set(f"'{name}' conflicts with a standard cut."); return
            cut_def = {"name": name, "col": col_name, "bin_type": bin_type,
                       "df_key": v_dfkey.get(), "min_n": v_min.get(), "enabled": True}
            if bin_type == "quantile":
                q_val = v_q.get(); lbl_str = v_labels.get().strip()
                cut_def["q"] = q_val
                if lbl_str:
                    lbls = [s.strip() for s in lbl_str.split(",")]
                    if len(lbls) != q_val:
                        err_var.set(f"Labels ({len(lbls)}) must match Q ({q_val})."); return
                    cut_def["labels"] = lbls
            elif bin_type == "fixed":
                try:
                    bins_val = [float(x.strip()) for x in v_bins_str.get().split(",")]
                    if len(bins_val) < 2: err_var.set("Need ≥2 bin edges."); return
                except ValueError:
                    err_var.set("Bin edges must be numbers."); return
                cut_def["bins"] = bins_val
                lbl_str = v_labels.get().strip()
                if lbl_str:
                    lbls = [s.strip() for s in lbl_str.split(",")]
                    expected = len(bins_val) - 1
                    if len(lbls) != expected:
                        err_var.set(f"Labels ({len(lbls)}) must equal bins–1 ({expected})."); return
                    cut_def["labels"] = lbls
            custom_cuts_data.append(cut_def)
            _refresh_tree(); _rebuild_custom_cc(); dlg.destroy()

        btn_r = ttk.Frame(dlg)
        btn_r.grid(row=r_d, column=0, columnspan=3, pady=(4, 8))
        ttk.Button(btn_r, text="Add",    command=_confirm).pack(side="left", padx=8)
        ttk.Button(btn_r, text="Cancel", command=dlg.destroy).pack(side="left", padx=8)

    def _remove_cut():
        sel = tree.selection()
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Select a cut to remove.", parent=root); return
        idx = tree.index(sel[0])
        removed_name = custom_cuts_data[idx].get("name", "")
        custom_cuts_data.pop(idx)
        for k in [k for k in list(cc_pair_vars.keys()) if removed_name in k]:
            del cc_pair_vars[k]
        _refresh_tree(); _rebuild_custom_cc()

    btn6_bar = ttk.Frame(f6)
    btn6_bar.grid(row=3, column=0, columnspan=5, sticky="w", padx=8, pady=4)
    ttk.Button(btn6_bar, text="+ Add Cut",        command=_add_cut_dialog).pack(side="left", padx=4)
    ttk.Button(btn6_bar, text="✕ Remove Selected", command=_remove_cut).pack(side="left",  padx=4)
    ttk.Label(
        f6,
        text="Column must exist in the input CSV.  "
             "categorical = raw string values.  "
             "quantile = bin a numeric column into N equal groups.  "
             "fixed = specify exact numeric bin edges.",
        foreground="#888",
    ).grid(row=4, column=0, columnspan=5, sticky="w", padx=8, pady=4)

    # -- Collect all settings into a config dict -------------------------------
    def _collect() -> dict:
        import copy
        c = copy.deepcopy(cfg)
        c.setdefault("data", {})
        c.setdefault("analysis", {})
        c.setdefault("preprocessing", {})
        c.setdefault("cuts", {})
        c.setdefault("cross_cuts", {})
        c.setdefault("feature_importance", {})
        c.setdefault("output", {})

        c["data"]["input_path"] = v_input.get().strip()
        c["data"]["output_dir"] = v_output.get().strip()

        c["analysis"]["alpha"]          = float(v_alpha.get())
        c["analysis"]["weight_col"]     = v_weight.get()
        c["analysis"]["min_n"]          = v_min_n.get()
        c["analysis"]["question_tags"]  = [qt for qt, v in qt_vars.items() if v.get()]
        c["analysis"]["mt_correction"]  = v_mt_corr.get()

        c["preprocessing"]["winsorize_lower"] = round(float(v_wlo.get()), 3)
        c["preprocessing"]["winsorize_upper"] = round(float(v_whi.get()), 3)
        c["preprocessing"]["freq_p99_cap"]    = v_freq_cap.get()

        for cut in ALL_CUTS:
            c["cuts"].setdefault(cut, {})
            c["cuts"][cut]["enabled"] = cut_en_vars[cut].get()
            c["cuts"][cut]["min_n"]   = cut_minn_vars[cut].get()

        c["cross_cuts"]["enabled"]          = v_cc_en.get()
        c["cross_cuts"]["min_n_multiplier"] = v_cc_mult.get()
        c["cross_cuts"]["pairs"]            = [p.split(" x ") for p, v in cc_pair_vars.items() if v.get()]

        c["feature_importance"]["enabled"]          = v_fi_en.get()
        c["feature_importance"]["stratify_by_tag"]   = v_fi_strat.get()
        c["feature_importance"]["n_estimators"]      = v_fi_nest.get()
        c["feature_importance"]["cv_folds"]          = int(v_fi_cv.get())
        c["feature_importance"]["top_n"]              = v_fi_topn.get()

        c["output"]["dpi"]          = int(v_dpi.get())
        c["output"]["save_figures"] = v_save_figs.get()
        c["output"]["export_csv"]   = v_export_csv.get()
        c["output"]["export_excel"] = v_export_xl.get()
        c["custom_cuts"]             = list(custom_cuts_data)
        c.setdefault("report", {})
        c["report"]["enabled"]    = v_rpt_en.get()
        # Empty string = use built-in key; actual key = use custom
        c["report"]["ai_api_key"] = "" if v_key_mode.get() == "builtin" else v_ai_key.get().strip()
        c["report"]["ai_model"]   = v_ai_model.get()
        return c

    # -- Progress window + background run -------------------------------------
    def _run_pipeline(c: dict) -> None:
        import matplotlib
        matplotlib.use("Agg")   # headless -- save to files, no display

        inp     = c["data"]["input_path"]
        out_dir = c["data"]["output_dir"]
        os.makedirs(out_dir, exist_ok=True)

        # Check PIL availability once
        try:
            from PIL import Image, ImageTk as _ITk
            _pil_ok = True
        except ImportError:
            _pil_ok = False

        prog = tk.Toplevel()
        prog.title("Pipeline Running...")
        prog.geometry("1200x660")
        prog.resizable(True, True)
        try: prog.attributes("-topmost", True)
        except Exception: pass

        ttk.Label(prog, text="BLS Meta-Analysis -- Pipeline Progress",
                  font=("", 11, "bold")).pack(padx=12, pady=(10, 4), anchor="w")

        # -- Split: log (left) + gallery (right) -------------------------------
        paned = ttk.PanedWindow(prog, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=2)

        # Left pane: log
        log_frame = ttk.Frame(paned)
        paned.add(log_frame, weight=2)
        log_box = scrolledtext.ScrolledText(
            log_frame, font=("Consolas", 10), wrap=tk.WORD, state="disabled",
            background="#1e1e2e", foreground="#cdd6f4")
        log_box.pack(fill="both", expand=True)

        # Right pane: gallery
        gal_frame = ttk.Frame(paned)
        paned.add(gal_frame, weight=3)
        ttk.Label(gal_frame, text="Generated Charts",
                  font=("", 10, "bold")).pack(anchor="w", padx=6, pady=(2, 0))
        if not _pil_ok:
            ttk.Label(gal_frame,
                      text="(Install Pillow to enable preview)",
                      foreground="#888").pack(anchor="w", padx=6)

        gal_canvas  = tk.Canvas(gal_frame, highlightthickness=0, background="#f5f5f5")
        gal_vbar    = ttk.Scrollbar(gal_frame, orient="vertical", command=gal_canvas.yview)
        gal_inner   = ttk.Frame(gal_canvas)
        gal_inner.bind(
            "<Configure>",
            lambda e: gal_canvas.configure(scrollregion=gal_canvas.bbox("all")))
        _gal_win = gal_canvas.create_window((0, 0), window=gal_inner, anchor="nw")
        gal_canvas.bind(
            "<Configure>",
            lambda e: gal_canvas.itemconfig(_gal_win, width=e.width))
        gal_canvas.configure(yscrollcommand=gal_vbar.set)
        gal_canvas.pack(side="left", fill="both", expand=True)
        gal_vbar.pack(side="right", fill="y")

        def _scroll_gal(event):
            if event.num == 4:   gal_canvas.yview_scroll(-1, "units")
            elif event.num == 5: gal_canvas.yview_scroll(1,  "units")
            else:                gal_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        gal_canvas.bind_all("<MouseWheel>", _scroll_gal)

        # Gallery state
        _photo_refs  = []          # keep PhotoImage refs alive
        _shown_files = set()
        _gal_pos     = [0, 0]      # [row, col]
        THUMB_W, COLS = 260, 2

        def _add_thumb(path: str) -> None:
            if not _pil_ok:
                return
            try:
                from PIL import Image, ImageTk
                img   = Image.open(path)
                ratio = THUMB_W / img.width
                img   = img.resize((THUMB_W, max(1, int(img.height * ratio))),
                                   Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                _photo_refs.append(photo)

                name = os.path.basename(path)
                cell = ttk.Frame(gal_inner, relief="ridge", borderwidth=1)
                cell.grid(row=_gal_pos[0], column=_gal_pos[1],
                          padx=4, pady=4, sticky="n")
                lbl = ttk.Label(cell, image=photo, cursor="hand2")
                lbl.pack()
                ttk.Label(cell, text=name, wraplength=THUMB_W,
                          font=("", 8)).pack(pady=(0, 2))

                def _open_full(p=path):
                    top = tk.Toplevel()
                    top.title(os.path.basename(p))
                    from PIL import Image, ImageTk
                    img2  = Image.open(p)
                    sw    = top.winfo_screenwidth()  * 0.85
                    sh    = top.winfo_screenheight() * 0.85
                    scale = min(sw / img2.width, sh / img2.height, 1.0)
                    img2  = img2.resize((max(1, int(img2.width * scale)),
                                         max(1, int(img2.height * scale))),
                                        Image.LANCZOS)
                    ph2 = ImageTk.PhotoImage(img2)
                    lbl2 = ttk.Label(top, image=ph2)
                    lbl2.image = ph2
                    lbl2.pack()

                lbl.bind("<Button-1>", lambda e: _open_full())

                _gal_pos[1] += 1
                if _gal_pos[1] >= COLS:
                    _gal_pos[1]  = 0
                    _gal_pos[0] += 1
                gal_canvas.configure(scrollregion=gal_canvas.bbox("all"))
            except Exception:
                pass  # silently skip broken images

        # File watcher: polls output dir for new PNGs
        _watcher_alive = [True]
        def _watch_output():
            import time
            while _watcher_alive[0]:
                try:
                    if os.path.exists(out_dir):
                        for fname in sorted(os.listdir(out_dir)):
                            if fname.lower().endswith(".png"):
                                fp = os.path.join(out_dir, fname)
                                if fp not in _shown_files:
                                    _shown_files.add(fp)
                                    prog.after(0, _add_thumb, fp)
                except Exception:
                    pass
                time.sleep(0.8)

        threading.Thread(target=_watch_output, daemon=True).start()

        def _on_close():
            _watcher_alive[0] = False
            prog.destroy()

        # -- Bottom bar --------------------------------------------------------
        bottom = ttk.Frame(prog)
        bottom.pack(fill="x", padx=8, pady=(2, 6))
        status_var  = tk.StringVar(value="Running...")
        _report_path = [None]
        ttk.Label(bottom, textvariable=status_var).pack(side="left", padx=4)
        btn_open_report = ttk.Button(
            bottom, text="📄 Open Report", state="disabled",
            command=lambda: webbrowser.open(
                "file:///" + _report_path[0].replace("\\", "/")))
        btn_open_report.pack(side="right", padx=4)
        btn_close = ttk.Button(bottom, text="Close", state="disabled",
                               command=_on_close)
        btn_close.pack(side="right", padx=4)
        prog.protocol("WM_DELETE_WINDOW", _on_close)

        def log(msg: str) -> None:
            log_box.configure(state="normal")
            log_box.insert(tk.END, msg + "\n")
            log_box.see(tk.END)
            log_box.configure(state="disabled")
            prog.update_idletasks()

        class _Tee(io.TextIOBase):
            def write(self, s):
                if s and s.strip():
                    prog.after(0, log, s.rstrip())
                return len(s)
            def flush(self): pass

        def _worker():
            tee = _Tee()
            try:
                with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                    import sys as _sys
                    for mod in list(_sys.modules.keys()):
                        if "bls_meta_pipeline" in mod or "report_generator" in mod:
                            del _sys.modules[mod]
                    import bls_meta_pipeline as _bpipe
                    _bpipe.OUT = out_dir

                    prog.after(0, log, "Loading data from:\n  " + inp)
                    df, df_product, df_objective = _bpipe.load_and_preprocess(inp)

                    prog.after(0, log, "\nInitialising pipeline...")
                    pipeline = _bpipe.BLSMetaPipeline(
                        df, df_product=df_product, df_objective=df_objective)
                    pipeline._apply_config(c)

                    # DEBUG: show what pairs will actually run
                    _dbg_pairs = c.get("cross_cuts", {}).get("pairs", [])
                    _dbg_enabled = c.get("cross_cuts", {}).get("enabled", True)
                    prog.after(0, log, f"\n[DEBUG] cross_cuts enabled={_dbg_enabled}, pairs from UI config: {_dbg_pairs}")
                    prog.after(0, log, f"[DEBUG] pipeline._cc_pairs after apply_config: {pipeline._cc_pairs}")

                    prog.after(0, log, "\nRunning all cuts...")
                    pipeline.full_run()   # report generation inside; path stored in pipeline

                    # Grab report path directly from pipeline (reliable vs. dir scan)
                    if pipeline._last_report_path:
                        _report_path[0] = pipeline._last_report_path

                prog.after(0, _done, True)
            except Exception:
                tb = traceback.format_exc()
                prog.after(0, log, "\nERROR:\n" + tb)
                prog.after(0, _done, False)

        def _done(ok: bool) -> None:
            _watcher_alive[0] = False   # stop polling
            # Final sweep to catch any last-written PNGs
            if os.path.exists(out_dir):
                for fname in sorted(os.listdir(out_dir)):
                    if fname.lower().endswith(".png"):
                        fp = os.path.join(out_dir, fname)
                        if fp not in _shown_files:
                            _shown_files.add(fp)
                            prog.after(0, _add_thumb, fp)
            if ok:
                if _report_path[0]:
                    status_var.set("Done!  Report: " + os.path.basename(_report_path[0]))
                    log("\nPipeline complete.\n  Charts & CSV: " + out_dir +
                        "\n  Report:       " + _report_path[0])
                    btn_open_report.configure(state="normal")
                else:
                    status_var.set("Done!  Results saved to: " + out_dir)
                    log("\nPipeline complete.  Charts & CSV saved to:\n  " + out_dir)
            else:
                status_var.set("Pipeline failed -- see log above.")
            btn_close.configure(state="normal")

        threading.Thread(target=_worker, daemon=True).start()

    # -- Bottom button bar -----------------------------------------------------
    btn_bar = ttk.Frame(root)
    btn_bar.pack(fill="x", side="bottom", padx=8, pady=8)

    def on_apply():
        c = _collect()
        if not c["data"]["input_path"]:
            messagebox.showerror("Missing input",
                                 "Set the input CSV in tab [1] Data Paths.")
            nb.select(0); return
        if not c["data"]["output_dir"]:
            messagebox.showerror("Missing output",
                                 "Set the output folder in tab [1] Data Paths.")
            nb.select(0); return
        if not os.path.exists(c["data"]["input_path"]):
            messagebox.showerror("File not found",
                                 f"Input file not found:\n{c['data']['input_path']}")
            nb.select(0); return
        _save_cfg(c)
        _run_pipeline(c)

    def on_save():
        c = _collect()
        _save_cfg(c)
        messagebox.showinfo("Saved", f"Config saved:\n{_CFG_PATH}")

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    ttk.Button(btn_bar, text="Apply & Run", command=on_apply).pack(side="left", padx=5)
    ttk.Button(btn_bar, text="Save Config",  command=on_save).pack(side="left",  padx=5)
    ttk.Button(btn_bar, text="Close",        command=root.destroy).pack(side="right", padx=5)
    ttk.Label(btn_bar,
              text="Apply & Run = save + load + run  |  Save Config = write config.yaml only",
              foreground="#888").pack(side="left", padx=10)


# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    build_app(root)
    root.mainloop()
