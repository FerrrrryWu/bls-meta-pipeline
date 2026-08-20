"""
BLS Meta-Analysis — Interactive Config UI
==========================================
Auto-detects environment and opens the appropriate UI:

  Jupyter / Google Colab  →  ipywidgets inline panel  (non-blocking)
  Local script            →  tkinter popup window      (blocking)

Usage (pipeline method — recommended)
--------------------------------------
    pipeline.launch_ui()
    # → UI appears; user adjusts settings; clicks "Apply & Run"
    # → pipeline._apply_config() is called, then pipeline.full_run()

Usage (standalone — config editor only)
-----------------------------------------
    from config_ui import launch_ui
    cfg = launch_ui()                # opens UI; returns config dict when done
    pipeline._apply_config(cfg)      # apply manually

Note for Colab
--------------
    ipywidgets is non-blocking. launch_ui() returns the *current* config
    immediately; the updated config is applied only when the user clicks a button.
    Use pipeline.launch_ui() (not the standalone function) for the full workflow.
"""
from __future__ import annotations
import copy, os
import yaml  # type: ignore

# ─── Metadata (must match CUTS_CONFIG in bls_meta_pipeline.py) ────────────────
ALL_CUTS = [
    "Watch Time", "Creative Count", "Video Duration", "VCR",
    "Frequency", "Weekly Impressions", "Objective", "Product Split",
    "Account Segment", "Audience Type", "Billing Type", "Spark Ads", "ACO",
]
ALL_QUESTION_TAGS = ["AD_RECALL", "AWARENESS", "INTENT"]
ALPHA_OPTIONS         = [0.01, 0.05, 0.10]
WEIGHT_OPTIONS        = ["iv_weight", "n_weight"]
DPI_OPTIONS           = [100, 150, 200, 300]
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

_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


# ─── Config I/O ───────────────────────────────────────────────────────────────
def _load_cfg(path: str = _CFG_PATH) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _save_cfg(cfg: dict, path: str = _CFG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ─── Environment detection ────────────────────────────────────────────────────
def _is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except ImportError:
        return False


# ─── Public entry point ───────────────────────────────────────────────────────
def launch_ui(config_path: str = _CFG_PATH, on_run=None) -> dict:
    """
    Open interactive Config UI. Returns the current config dict.

    Parameters
    ----------
    config_path : str
        Path to config.yaml (default: same directory as this file).
    on_run : callable(cfg: dict) | None
        Called with the final config dict when the user clicks "Apply & Run".
    """
    cfg = _load_cfg(config_path)
    if _is_notebook():
        return _notebook_ui(cfg, config_path, on_run)
    else:
        return _tkinter_ui(cfg, config_path, on_run)


# ═══════════════════════════════════════════════════════════════════════════════
#  NOTEBOOK UI  (ipywidgets)
# ═══════════════════════════════════════════════════════════════════════════════

def _notebook_ui(cfg: dict, config_path: str, on_run) -> dict:
    try:
        import ipywidgets as w                              # type: ignore
        from IPython.display import display, clear_output  # type: ignore
    except ImportError:
        print("❌  ipywidgets not installed.  Run:  pip install ipywidgets")
        return cfg

    result = {"config": copy.deepcopy(cfg)}

    ana        = cfg.get("analysis",           {})
    pre        = cfg.get("preprocessing",      {})
    cuts       = cfg.get("cuts",               {})
    cc         = cfg.get("cross_cuts",         {})
    fi         = cfg.get("feature_importance", {})
    out        = cfg.get("output",             {})
    adv_filter = cfg.get("advertiser_filter",  {})

    # ── Layout helpers ────────────────────────────────────────────────────────
    W = w.Layout
    lbl_style = {"description_width": "140px"}

    def row(*items, gap="10px"):
        return w.HBox(list(items), layout=W(align_items="center", gap=gap))

    def hint(text):
        return w.HTML(f"<span style='color:#888;font-size:0.85em'>{text}</span>")

    def section(title):
        return w.HTML(
            f"<h3 style='margin:10px 0 2px;padding-bottom:4px;"
            f"border-bottom:1px solid #ddd;color:#333'>{title}</h3>"
        )

    # ── Tab 1: Analysis + Preprocessing ─────────────────────────────────────
    w_alpha = w.BoundedFloatText(
        value=ana.get("alpha", 0.10),
        min=0.001, max=0.999, step=0.01,
        description="Alpha:", style=lbl_style, layout=W(width="200px"),
    )
    w_mt_corr = w.Dropdown(
        options=MT_CORRECTION_OPTIONS,
        value=ana.get("mt_correction", "none"),
        description="MT Correction:", style=lbl_style, layout=W(width="230px"),
    )
    w_weight = w.Dropdown(
        options=WEIGHT_OPTIONS, value=ana.get("weight_col", "iv_weight"),
        description="Weight col:", style=lbl_style, layout=W(width="230px"),
    )
    w_min_n = w.BoundedIntText(
        value=ana.get("min_n", 5), min=2, max=50,
        description="Global min_n:", style=lbl_style, layout=W(width="210px"),
    )
    w_qtags = w.SelectMultiple(
        options=ALL_QUESTION_TAGS,
        value=list(ana.get("question_tags", ALL_QUESTION_TAGS)),
        rows=4, description="Metrics:", style=lbl_style, layout=W(width="320px"),
    )
    w_wlo = w.FloatSlider(
        value=pre.get("winsorize_lower", 0.02), min=0.01, max=0.05, step=0.01,
        readout_format=".2f", description="Winsorize lo:",
        style=lbl_style, layout=W(width="400px"),
    )
    w_whi = w.FloatSlider(
        value=pre.get("winsorize_upper", 0.98), min=0.95, max=0.99, step=0.01,
        readout_format=".2f", description="Winsorize hi:",
        style=lbl_style, layout=W(width="400px"),
    )
    w_freq_cap = w.Checkbox(
        value=pre.get("freq_p99_cap", True),
        description="Cap frequency at P99 before binning",
    )

    # ── Advertiser Filter widgets ──────────────────────────────────────────
    w_adv_en = w.Checkbox(
        value=adv_filter.get("enabled", False),
        description="Filter by Advertiser IDs  (unchecked = all advertisers)",
        layout=W(width="500px"),
    )
    w_adv_ids = w.Textarea(
        value=", ".join(str(i) for i in adv_filter.get("ids", [])),
        placeholder="Enter comma-separated advertiser IDs, e.g.: 123456, 789012",
        layout=W(width="500px", height="70px"),
    )
    w_adv_col = w.Text(
        value=adv_filter.get("id_column", "advertiser_id"),
        description="ID column:",
        style=lbl_style,
        layout=W(width="340px"),
    )

    tab1 = w.VBox([
        section("⚙️ Analysis"),
        row(w_alpha,    hint("0.01 strict | 0.05 standard | 0.10 lenient")),
        row(w_mt_corr,  hint("BH = FDR control (recommended) | bonferroni = conservative | holm = stepwise")),
        row(w_weight,   hint("iv_weight = 1/SE²  &nbsp;·&nbsp;  n_weight = harmonic N")),
        row(w_min_n,  hint("min campaigns per group bin  (2–50)")),
        w.HTML("<br><b>Question Tags</b>&nbsp;"
               "<small style='color:#888'>Ctrl / Cmd + click for multi-select</small>"),
        w_qtags,
        section("🧹 Preprocessing"),
        row(w_wlo, hint("lower lift clip percentile  (0.01–0.05)")),
        row(w_whi, hint("upper lift clip percentile  (0.95–0.99)")),
        w_freq_cap,
        section("🔍 Advertiser Filter"),
        w_adv_en,
        row(w_adv_col, hint("column name in your CSV that holds advertiser / account IDs")),
        w.HTML("<b>Advertiser IDs</b>&nbsp;<small style='color:#888'>"
               "comma-separated; leave empty for all</small>"),
        w_adv_ids,
    ], layout=W(padding="12px"))

    # ── Tab 2: Cuts ──────────────────────────────────────────────────────────
    cut_en_w   = {}
    cut_minn_w = {}
    header_row = w.HBox([
        w.HTML("<b style='width:210px;display:inline-block'>Cut</b>"),
        w.HTML("<b style='width:80px;display:inline-block'>Enabled</b>"),
        w.HTML("<b>min_n override</b>"),
    ])
    cut_rows = [section("✂️ Cut Settings"), header_row]
    for cut in ALL_CUTS:
        cut_cfg = cuts.get(cut, {})
        en = w.Checkbox(value=cut_cfg.get("enabled", True),
                        layout=W(width="80px"))
        mn = w.BoundedIntText(
            value=cut_cfg.get("min_n", ana.get("min_n", 5)),
            min=2, max=50, layout=W(width="90px"),
        )
        cut_en_w[cut]   = en
        cut_minn_w[cut] = mn
        cut_rows.append(w.HBox([
            w.HTML(f"<span style='width:210px;display:inline-block'>{cut}</span>"),
            en, mn,
        ]))

    tab2 = w.VBox(cut_rows, layout=W(padding="12px"))

    # ── Tab 3: Cross-Cuts ────────────────────────────────────────────────────
    w_cc_en = w.Checkbox(
        value=cc.get("enabled", True),
        description="Enable cross-cut analysis",
    )
    w_cc_mult = w.BoundedIntText(
        value=cc.get("min_n_multiplier", 3), min=2, max=10,
        description="min_n ×:", style=lbl_style, layout=W(width="200px"),
    )
    _tier1_strs = {f"{p[0]} × {p[1]}" for p in TIER1_PAIRS}
    existing_pairs = {
        f"{p[0]} × {p[1]}"
        for p in cc.get("pairs", list(TIER1_PAIRS))
    } or _tier1_strs

    tier1_rows, tier2_rows = [], []
    cc_pair_widgets = {}
    for pairs_list, rows_list in [(TIER1_PAIRS, tier1_rows), (TIER2_PAIRS, tier2_rows)]:
        for pair in pairs_list:
            pair_str = f"{pair[0]} × {pair[1]}"
            cb = w.Checkbox(
                value=(pair_str in existing_pairs),
                description=pair_str,
                layout=W(width="340px"),
            )
            cc_pair_widgets[pair_str] = cb
            rows_list.append(cb)

    tab3 = w.VBox([
        section("🔀 Cross-Cut Analysis"),
        w_cc_en,
        row(w_cc_mult, hint("cross min_n = global_min_n × this  (range 2–10)")),
        w.HTML("<br><b>⭐ Tier 1 — Recommended (default on)</b>"),
        *tier1_rows,
        w.HTML("<br><b>◎ Tier 2 — Optional (default off)</b>"),
        *tier2_rows,
    ], layout=W(padding="12px"))

    # ── Tab 4: Feature Importance + Output ───────────────────────────────────
    w_fi_en = w.Checkbox(
        value=fi.get("enabled", True),
        description="Enable RF feature importance",
    )
    w_fi_strat = w.Checkbox(
        value=fi.get("stratify_by_tag", False),
        description="Stratify FI by question tag (separate RF per tag, comparison plot)",
        layout=W(width="500px"),
    )
    w_fi_nest = w.IntSlider(
        value=fi.get("n_estimators", 300), min=100, max=500, step=50,
        description="RF trees:", style=lbl_style, layout=W(width="420px"),
    )
    w_fi_cv = w.Dropdown(
        options=[(str(v), v) for v in CV_FOLD_OPTIONS],
        value=fi.get("cv_folds", 5),
        description="CV folds:", style=lbl_style, layout=W(width="200px"),
    )
    w_fi_topn = w.IntSlider(
        value=fi.get("top_n", 15), min=5, max=30, step=1,
        description="Top N features:", style=lbl_style, layout=W(width="420px"),
    )
    w_dpi = w.Dropdown(
        options=[(str(v), v) for v in DPI_OPTIONS],
        value=out.get("dpi", 150),
        description="Figure DPI:", style=lbl_style, layout=W(width="200px"),
    )
    w_save_figs  = w.Checkbox(value=out.get("save_figures", True),  description="Save PNG figures")
    w_export_csv = w.Checkbox(value=out.get("export_csv",   True),  description="Export CSV")
    w_export_xl  = w.Checkbox(value=out.get("export_excel", False), description="Export Excel (.xlsx)")

    tab4 = w.VBox([
        section("🌲 Feature Importance"),
        w_fi_en,
        w_fi_strat,
        row(w_fi_nest, hint("100–500  (more trees = more stable, but slower)")),
        row(w_fi_cv,   hint("options: 3 | 5 | 10")),
        row(w_fi_topn, hint("top N features to show  (5–30)")),
        section("📤 Output"),
        row(w_dpi,     hint("100 draft | 150 default | 300 print-quality")),
        w_save_figs, w_export_csv, w_export_xl,
    ], layout=W(padding="12px"))

    # ── Tabs container ────────────────────────────────────────────────────────
    tabs = w.Tab(children=[tab1, tab2, tab3, tab4])
    try:
        tabs.titles = ["⚙️ Analysis", "✂️ Cuts", "🔀 Cross-Cuts", "🌲 FI & Output"]
    except AttributeError:
        for i, t in enumerate(["⚙️ Analysis", "✂️ Cuts", "🔀 Cross-Cuts", "🌲 FI & Output"]):
            tabs.set_title(i, t)

    # ── Collect helper ────────────────────────────────────────────────────────
    def _collect() -> dict:
        c = copy.deepcopy(cfg)
        c.setdefault("analysis",           {})
        c.setdefault("preprocessing",      {})
        c.setdefault("cuts",               {})
        c.setdefault("cross_cuts",         {})
        c.setdefault("feature_importance", {})
        c.setdefault("output",             {})

        c["analysis"]["alpha"]          = w_alpha.value
        c["analysis"]["weight_col"]     = w_weight.value
        c["analysis"]["min_n"]          = w_min_n.value
        c["analysis"]["question_tags"]  = list(w_qtags.value)
        c["analysis"]["mt_correction"]  = w_mt_corr.value

        c["preprocessing"]["winsorize_lower"] = round(w_wlo.value,  3)
        c["preprocessing"]["winsorize_upper"] = round(w_whi.value,  3)
        c["preprocessing"]["freq_p99_cap"]    = w_freq_cap.value

        for cut in ALL_CUTS:
            c["cuts"].setdefault(cut, {})
            c["cuts"][cut]["enabled"] = cut_en_w[cut].value
            c["cuts"][cut]["min_n"]   = cut_minn_w[cut].value

        c["cross_cuts"]["enabled"]          = w_cc_en.value
        c["cross_cuts"]["min_n_multiplier"] = w_cc_mult.value
        c["cross_cuts"]["pairs"]            = [
            p.split(" × ") for p, cb in cc_pair_widgets.items() if cb.value
        ]

        c["feature_importance"]["enabled"]          = w_fi_en.value
        c["feature_importance"]["stratify_by_tag"]   = w_fi_strat.value
        c["feature_importance"]["n_estimators"]      = w_fi_nest.value
        c["feature_importance"]["cv_folds"]          = w_fi_cv.value
        c["feature_importance"]["top_n"]              = w_fi_topn.value

        c["output"]["dpi"]          = w_dpi.value
        c["output"]["save_figures"] = w_save_figs.value
        c["output"]["export_csv"]   = w_export_csv.value
        c["output"]["export_excel"] = w_export_xl.value

        c.setdefault("advertiser_filter", {})
        c["advertiser_filter"]["enabled"]   = w_adv_en.value
        c["advertiser_filter"]["ids"]       = [
            x.strip() for x in w_adv_ids.value.split(",") if x.strip()
        ]
        c["advertiser_filter"]["id_column"] = w_adv_col.value
        return c

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_run    = w.Button(description="🚀 Apply & Run", button_style="success",
                           layout=W(width="160px", height="36px"))
    btn_save   = w.Button(description="💾 Save Config", button_style="info",
                           layout=W(width="160px", height="36px"))
    btn_cancel = w.Button(description="✖ Cancel",       button_style="danger",
                           layout=W(width="120px", height="36px"))
    out_log    = w.Output()

    def _on_run(b):
        c = _collect()
        result["config"] = c
        with out_log:
            clear_output()
            print("✅  Config applied!  Starting pipeline run …")
        if on_run:
            on_run(c)

    def _on_save(b):
        c = _collect()
        result["config"] = c
        _save_cfg(c, config_path)
        with out_log:
            clear_output()
            print(f"💾  Config saved → {config_path}")

    def _on_cancel(b):
        with out_log:
            clear_output()
            print("✖  Cancelled — no changes applied.")

    btn_run.on_click(_on_run)
    btn_save.on_click(_on_save)
    btn_cancel.on_click(_on_cancel)

    panel = w.VBox([
        w.HTML("<h2 style='margin-bottom:6px'>🔬  BLS Meta-Analysis — Config UI</h2>"),
        tabs,
        w.HBox([btn_run, btn_save, btn_cancel], layout=W(margin="10px 0 4px", gap="8px")),
        out_log,
    ])
    display(panel)
    return result["config"]


# ═══════════════════════════════════════════════════════════════════════════════
#  TKINTER UI  (local script)
# ═══════════════════════════════════════════════════════════════════════════════

def _tkinter_ui(cfg: dict, config_path: str, on_run) -> dict:
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError:
        print("❌  tkinter not available on this system.")
        return cfg

    result = {"config": copy.deepcopy(cfg), "ok": False}

    ana        = cfg.get("analysis",           {})
    pre        = cfg.get("preprocessing",      {})
    cuts_cfg   = cfg.get("cuts",               {})
    cc         = cfg.get("cross_cuts",         {})
    fi         = cfg.get("feature_importance", {})
    out        = cfg.get("output",             {})
    adv_filter = cfg.get("advertiser_filter",  {})

    # ── Root window ───────────────────────────────────────────────────────────
    root = tk.Tk()
    root.title("BLS Meta-Analysis — Config UI")
    root.geometry("860x700")
    root.resizable(True, True)
    try:
        root.tk.call("tk", "scaling", 1.15)
    except Exception:
        pass

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # ── Scrollable frame helper ───────────────────────────────────────────────
    def scrollable(parent) -> ttk.Frame:
        cont   = ttk.Frame(parent)
        cont.pack(fill="both", expand=True)
        canvas = tk.Canvas(cont, highlightthickness=0)
        vbar   = ttk.Scrollbar(cont, orient="vertical", command=canvas.yview)
        inner  = ttk.Frame(canvas)
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(win_id, width=e.width),
        )
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        def _scroll(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
            else:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _scroll)
        canvas.bind_all("<Button-4>",   _scroll)
        canvas.bind_all("<Button-5>",   _scroll)
        return inner

    # ── Small helpers ─────────────────────────────────────────────────────────
    def heading(parent, text, row):
        ttk.Label(parent, text=text, font=("", 10, "bold")).grid(
            row=row, column=0, columnspan=4,
            sticky="w", padx=8, pady=(10, 2),
        )

    def hint(parent, text, row, col=3):
        ttk.Label(parent, text=text, foreground="#777").grid(
            row=row, column=col, sticky="w", padx=4,
        )

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    # ─── TAB 1: Analysis + Preprocessing ─────────────────────────────────────
    tab1 = ttk.Frame(nb);  nb.add(tab1, text="⚙️ Analysis")
    f1   = scrollable(tab1)
    heading(f1, "Analysis Settings", 0)

    r = 1
    ttk.Label(f1, text="Alpha:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_alpha = tk.StringVar(value=str(ana.get("alpha", 0.10)))
    ttk.Combobox(f1, textvariable=v_alpha,
                 values=[str(v) for v in ALPHA_OPTIONS],
                 width=8, state="normal").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "0.01 strict  |  0.05 standard  |  0.10 lenient  (或手动输入)", r); r += 1

    ttk.Label(f1, text="MT Correction:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_mt_corr = tk.StringVar(value=ana.get("mt_correction", "none"))
    ttk.Combobox(f1, textvariable=v_mt_corr, values=MT_CORRECTION_OPTIONS,
                 width=12, state="readonly").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "BH = FDR control (recommended)  |  bonferroni = conservative  |  holm = stepwise", r); r += 1

    ttk.Label(f1, text="Weight col:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_weight = tk.StringVar(value=ana.get("weight_col", "iv_weight"))
    ttk.Combobox(f1, textvariable=v_weight,
                 values=WEIGHT_OPTIONS, width=14, state="readonly").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "iv_weight = 1/SE²  |  n_weight = harmonic N", r); r += 1

    ttk.Label(f1, text="Global min_n:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_min_n = tk.IntVar(value=ana.get("min_n", 5))
    ttk.Spinbox(f1, from_=2, to=50, textvariable=v_min_n,
                width=6).grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "min campaigns per group bin  (range 2–50)", r); r += 1

    ttk.Label(f1, text="Question Tags:", font=("", 9, "bold")).grid(
        row=r, column=0, sticky="nw", padx=8, pady=3)
    active_tags = ana.get("question_tags", ALL_QUESTION_TAGS)
    qt_vars = {}
    for qt in ALL_QUESTION_TAGS:
        v = tk.BooleanVar(value=(qt in active_tags))
        ttk.Checkbutton(f1, text=qt, variable=v).grid(
            row=r, column=1, sticky="w", padx=4, pady=1)
        qt_vars[qt] = v
        r += 1

    ttk.Separator(f1, orient="horizontal").grid(
        row=r, columnspan=4, sticky="ew", padx=8, pady=6); r += 1
    heading(f1, "Preprocessing", r); r += 1

    ttk.Label(f1, text="Winsorize lower:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_wlo = tk.DoubleVar(value=pre.get("winsorize_lower", 0.02))
    ttk.Spinbox(f1, from_=0.01, to=0.05, increment=0.01,
                textvariable=v_wlo, width=7, format="%.2f").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "lower lift clip percentile  (0.01–0.05)", r); r += 1

    ttk.Label(f1, text="Winsorize upper:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_whi = tk.DoubleVar(value=pre.get("winsorize_upper", 0.98))
    ttk.Spinbox(f1, from_=0.95, to=0.99, increment=0.01,
                textvariable=v_whi, width=7, format="%.2f").grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "upper lift clip percentile  (0.95–0.99)", r); r += 1

    v_freq_cap = tk.BooleanVar(value=pre.get("freq_p99_cap", True))
    ttk.Checkbutton(f1, text="Cap frequency at P99 before binning",
                    variable=v_freq_cap).grid(
        row=r, column=0, columnspan=3, sticky="w", padx=8, pady=3); r += 1

    # ── Advertiser Filter ────────────────────────────────────────────────
    ttk.Separator(f1, orient="horizontal").grid(
        row=r, columnspan=4, sticky="ew", padx=8, pady=6); r += 1
    heading(f1, "🔍 Advertiser Filter", r); r += 1

    v_adv_en = tk.BooleanVar(value=adv_filter.get("enabled", False))
    ttk.Checkbutton(
        f1, text="Filter by Advertiser IDs  (unchecked = all advertisers)",
        variable=v_adv_en,
    ).grid(row=r, column=0, columnspan=3, sticky="w", padx=8, pady=3); r += 1

    ttk.Label(f1, text="Advertiser IDs:").grid(row=r, column=0, sticky="nw", padx=8, pady=3)
    adv_text = tk.Text(f1, height=3, width=50)
    adv_text.insert("1.0", ", ".join(str(i) for i in adv_filter.get("ids", [])))
    adv_text.grid(row=r, column=1, columnspan=2, sticky="w", padx=4, pady=3)
    hint(f1, "comma-separated advertiser IDs; leave empty for all", r); r += 1

    ttk.Label(f1, text="ID column:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
    v_adv_col = tk.StringVar(value=adv_filter.get("id_column", "advertiser_id"))
    ttk.Entry(f1, textvariable=v_adv_col, width=25).grid(row=r, column=1, sticky="w", padx=4)
    hint(f1, "CSV column name containing advertiser/account IDs", r); r += 1

    # ─── TAB 2: Cuts ─────────────────────────────────────────────────────────
    tab2 = ttk.Frame(nb);  nb.add(tab2, text="✂️ Cuts")
    f2   = scrollable(tab2)
    ttk.Label(f2, text="Cut Settings", font=("", 10, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 2))
    for txt, col in [("Cut Name", 0), ("Enabled", 1), ("min_n", 2)]:
        ttk.Label(f2, text=txt, font=("", 9, "bold")).grid(row=1, column=col, padx=10)

    cut_en_vars   = {}
    cut_minn_vars = {}
    for idx, cut in enumerate(ALL_CUTS):
        cc_c = cuts_cfg.get(cut, {})
        r2   = idx + 2
        ttk.Label(f2, text=cut).grid(row=r2, column=0, sticky="w", padx=8, pady=2)
        v_en = tk.BooleanVar(value=cc_c.get("enabled", True))
        v_mn = tk.IntVar(value=cc_c.get("min_n", ana.get("min_n", 5)))
        ttk.Checkbutton(f2, variable=v_en).grid(row=r2, column=1, padx=10)
        ttk.Spinbox(f2, from_=2, to=50, textvariable=v_mn, width=6).grid(row=r2, column=2, padx=10)
        cut_en_vars[cut]   = v_en
        cut_minn_vars[cut] = v_mn

    # ─── TAB 3: Cross-Cuts ────────────────────────────────────────────────────
    tab3 = ttk.Frame(nb);  nb.add(tab3, text="🔀 Cross-Cuts")
    f3   = scrollable(tab3)
    ttk.Label(f3, text="Cross-Cut Analysis", font=("", 10, "bold")).grid(
        row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 2))

    v_cc_en = tk.BooleanVar(value=cc.get("enabled", True))
    ttk.Checkbutton(f3, text="Enable cross-cut analysis",
                    variable=v_cc_en).grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=3)

    ttk.Label(f3, text="min_n multiplier:").grid(row=2, column=0, sticky="w", padx=8, pady=3)
    v_cc_mult = tk.IntVar(value=cc.get("min_n_multiplier", 3))
    ttk.Spinbox(f3, from_=2, to=10, textvariable=v_cc_mult, width=6).grid(
        row=2, column=1, sticky="w", padx=4)
    ttk.Label(f3, text="cross min_n = global_min_n × this  (range 2–10)",
              foreground="#777").grid(row=2, column=2, columnspan=2, sticky="w", padx=4)

    ttk.Label(f3, text="Select cross-cut pairs:", font=("", 9, "bold")).grid(
        row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(8, 2))
    ttk.Label(f3, text="(check pairs to analyze as 2-D heatmaps)",
              foreground="#777").grid(row=4, column=0, columnspan=4, sticky="w", padx=8)

    _t1_strs = {f"{p[0]} × {p[1]}" for p in TIER1_PAIRS}
    existing_pairs_set = (
        {f"{p[0]} × {p[1]}" for p in cc.get("pairs", list(TIER1_PAIRS))}
        or _t1_strs
    )
    cc_pair_vars = {}
    r3 = 5

    ttk.Label(f3, text="⭐ Tier 1 — Recommended (default on):",
              font=("", 9, "bold")).grid(
        row=r3, column=0, columnspan=4, sticky="w", padx=8, pady=(6, 2)); r3 += 1
    for pair in TIER1_PAIRS:
        pair_str = f"{pair[0]} × {pair[1]}"
        v_pair   = tk.BooleanVar(value=(pair_str in existing_pairs_set))
        ttk.Checkbutton(f3, text=pair_str, variable=v_pair).grid(
            row=r3, column=0, columnspan=4, sticky="w", padx=20, pady=1)
        cc_pair_vars[pair_str] = v_pair; r3 += 1

    ttk.Label(f3, text="◎ Tier 2 — Optional (default off):",
              font=("", 9, "bold")).grid(
        row=r3, column=0, columnspan=4, sticky="w", padx=8, pady=(10, 2)); r3 += 1
    for pair in TIER2_PAIRS:
        pair_str = f"{pair[0]} × {pair[1]}"
        v_pair   = tk.BooleanVar(value=(pair_str in existing_pairs_set))
        ttk.Checkbutton(f3, text=pair_str, variable=v_pair).grid(
            row=r3, column=0, columnspan=4, sticky="w", padx=20, pady=1)
        cc_pair_vars[pair_str] = v_pair; r3 += 1

    # ─── TAB 4: Feature Importance + Output ──────────────────────────────────
    tab4 = ttk.Frame(nb);  nb.add(tab4, text="🌲 FI & Output")
    f4   = scrollable(tab4)
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
    ttk.Spinbox(f4, from_=100, to=500, increment=50,
                textvariable=v_fi_nest, width=8).grid(row=r4, column=1, sticky="w", padx=4)
    hint(f4, "100–500  (more = stable, slower)", r4); r4 += 1

    ttk.Label(f4, text="CV folds:").grid(row=r4, column=0, sticky="w", padx=8, pady=3)
    v_fi_cv = tk.StringVar(value=str(fi.get("cv_folds", 5)))
    ttk.Combobox(f4, textvariable=v_fi_cv,
                 values=[str(v) for v in CV_FOLD_OPTIONS],
                 width=6, state="readonly").grid(row=r4, column=1, sticky="w", padx=4)
    hint(f4, "options: 3 | 5 | 10", r4); r4 += 1

    ttk.Label(f4, text="Top N features:").grid(row=r4, column=0, sticky="w", padx=8, pady=3)
    v_fi_topn = tk.IntVar(value=fi.get("top_n", 15))
    ttk.Spinbox(f4, from_=5, to=30, textvariable=v_fi_topn,
                width=6).grid(row=r4, column=1, sticky="w", padx=4)
    hint(f4, "5–30", r4); r4 += 1

    ttk.Separator(f4, orient="horizontal").grid(
        row=r4, columnspan=4, sticky="ew", padx=8, pady=6); r4 += 1
    heading(f4, "Output Settings", r4); r4 += 1

    ttk.Label(f4, text="Figure DPI:").grid(row=r4, column=0, sticky="w", padx=8, pady=3)
    v_dpi = tk.StringVar(value=str(out.get("dpi", 150)))
    ttk.Combobox(f4, textvariable=v_dpi,
                 values=[str(v) for v in DPI_OPTIONS],
                 width=6, state="readonly").grid(row=r4, column=1, sticky="w", padx=4)
    hint(f4, "100 draft  |  150 default  |  300 print-quality", r4); r4 += 1

    v_save_figs  = tk.BooleanVar(value=out.get("save_figures", True))
    v_export_csv = tk.BooleanVar(value=out.get("export_csv",   True))
    v_export_xl  = tk.BooleanVar(value=out.get("export_excel", False))
    for txt, var in [
        ("Save PNG figures to output dir",   v_save_figs),
        ("Export results as CSV",            v_export_csv),
        ("Export results as Excel (.xlsx)",  v_export_xl),
    ]:
        ttk.Checkbutton(f4, text=txt, variable=var).grid(
            row=r4, column=0, columnspan=3, sticky="w", padx=8, pady=2); r4 += 1

    # ─── Collect ──────────────────────────────────────────────────────────────
    def _collect() -> dict:
        c = copy.deepcopy(cfg)
        c.setdefault("analysis",           {})
        c.setdefault("preprocessing",      {})
        c.setdefault("cuts",               {})
        c.setdefault("cross_cuts",         {})
        c.setdefault("feature_importance", {})
        c.setdefault("output",             {})

        try:
            _alpha_val = float(v_alpha.get())
            if not (0 < _alpha_val < 1):
                raise ValueError
            c["analysis"]["alpha"] = _alpha_val
        except ValueError:
            messagebox.showerror("Invalid Alpha", "Alpha must be a number between 0 and 1 (e.g. 0.05)")
            return
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
        c["cross_cuts"]["pairs"]            = [
            p.split(" × ") for p, v in cc_pair_vars.items() if v.get()
        ]

        c["feature_importance"]["enabled"]          = v_fi_en.get()
        c["feature_importance"]["stratify_by_tag"]   = v_fi_strat.get()
        c["feature_importance"]["n_estimators"]      = v_fi_nest.get()
        c["feature_importance"]["cv_folds"]          = int(v_fi_cv.get())
        c["feature_importance"]["top_n"]              = v_fi_topn.get()

        c["output"]["dpi"]          = int(v_dpi.get())
        c["output"]["save_figures"] = v_save_figs.get()
        c["output"]["export_csv"]   = v_export_csv.get()
        c["output"]["export_excel"] = v_export_xl.get()

        c.setdefault("advertiser_filter", {})
        c["advertiser_filter"]["enabled"]   = v_adv_en.get()
        c["advertiser_filter"]["ids"]       = [
            x.strip() for x in adv_text.get("1.0", "end").split(",") if x.strip()
        ]
        c["advertiser_filter"]["id_column"] = v_adv_col.get()
        return c

    # ─── Bottom buttons ───────────────────────────────────────────────────────
    btn_bar = ttk.Frame(root)
    btn_bar.pack(fill="x", side="bottom", padx=8, pady=8)

    def on_apply():
        c = _collect()
        result.update(config=c, ok=True)
        root.destroy()
        if on_run:
            on_run(c)

    def on_save():
        c = _collect()
        result["config"] = c
        _save_cfg(c, config_path)
        messagebox.showinfo("Saved", f"Config saved →\n{config_path}")

    def on_cancel():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_cancel)

    ttk.Button(btn_bar, text="🚀 Apply & Run", command=on_apply).pack(side="left", padx=5)
    ttk.Button(btn_bar, text="💾 Save Config",  command=on_save).pack(side="left",  padx=5)
    ttk.Button(btn_bar, text="✖ Cancel",        command=on_cancel).pack(side="right", padx=5)
    ttk.Label(
        btn_bar,
        text="Apply & Run = apply + start pipeline  ·  Save Config = write config.yaml  ·  Cancel = discard",
        foreground="#888",
    ).pack(side="left", padx=10)

    root.mainloop()
    return result["config"]
