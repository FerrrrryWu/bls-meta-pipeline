"""
BLS Meta-Analysis Pipeline
===========================
Standalone end-to-end tool for TikTok Brand Lift Study meta-analysis.

Usage (script):
    python bls_meta_pipeline.py

Usage (import):
    from bls_meta_pipeline import BLSMetaPipeline, load_and_preprocess

    df, df_product, df_objective = load_and_preprocess("path/to/data.csv")
    pipeline = BLSMetaPipeline(df, df_product=df_product, df_objective=df_objective)
    pipeline.full_run()

Config:
    Edit config.yaml to set DATA path, OUT path, and analysis parameters.
    Or pass config= dict to BLSMetaPipeline() to override at runtime.
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
import os
import re
import ast
import yaml
import warnings
import textwrap

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import norm as scipy_norm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ─── Load Config ──────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def _load_config(path=_CONFIG_PATH):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

_CFG = _load_config()

DATA = _CFG.get("data", {}).get("input_path", "Meta_analysis.csv")
OUT  = _CFG.get("data", {}).get("output_dir",  "output/")
os.makedirs(OUT, exist_ok=True)

_ANA      = _CFG.get("analysis",      {})
_PRE      = _CFG.get("preprocessing", {})
_FI       = _CFG.get("feature_importance", {})

DEFAULT_ALPHA        = _ANA.get("alpha",        0.10)
DEFAULT_MIN_N        = _ANA.get("min_n",        5)
DEFAULT_WEIGHT_COL   = _ANA.get("weight_col",   "iv_weight")
DEFAULT_QUESTION_TAGS = _ANA.get("question_tags",
                                  ["AD_RECALL", "AWARENESS", "INTENT"])
MT_CORRECTION         = _ANA.get("mt_correction", "none")  # none | bh | bonferroni | holm

WINSORIZE_LO  = _PRE.get("winsorize_lower", 0.02)
WINSORIZE_HI  = _PRE.get("winsorize_upper", 0.98)
FREQ_P99_CAP  = _PRE.get("freq_p99_cap",    True)

RF_N_ESTIMATORS = _FI.get("n_estimators", 300)
RF_CV_FOLDS     = _FI.get("cv_folds",     5)
RF_TOP_N        = _FI.get("top_n",        15)

# ─── Extended Config ──────────────────────────────────────────────────────────
_CUTS_YAML = _CFG.get("cuts",        {})
_CC_YAML   = _CFG.get("cross_cuts",  {})
_OUT_YAML  = _CFG.get("output",      {})

FIGURE_DPI           = _OUT_YAML.get("dpi",           150)
SAVE_FIGS_DEFAULT    = _OUT_YAML.get("save_figures",   True)
EXPORT_CSV_DEFAULT   = _OUT_YAML.get("export_csv",     True)
EXPORT_EXCEL_DEFAULT = _OUT_YAML.get("export_excel",   False)
FI_ENABLED           = _CFG.get("feature_importance", {}).get("enabled", True)
CC_ENABLED           = _CC_YAML.get("enabled",         True)

RECOMMENDED_PAIRS = {
    "tier1": [
        ["Watch Time",     "VCR"],
        ["Watch Time",     "Objective"],
        ["Video Duration", "VCR"],
        ["Creative Count", "Objective"],
        ["Frequency",      "Audience Type"],
    ],
    "tier2": [
        ["Watch Time",     "Creative Count"],
        ["Billing Type",   "Objective"],
        ["Spark Ads",      "Audience Type"],
        ["ACO",            "Creative Count"],
        ["VCR",            "Frequency"],
    ],
}
CC_DEFAULT_PAIRS = _CC_YAML.get("pairs", RECOMMENDED_PAIRS["tier1"])


# ─── Grouping Maps ────────────────────────────────────────────────────────────

OBJ_MAP = {
    # Brand Auction
    "Video Views (Auction)": "Brand Auction",
    "Reach (Auction)":       "Brand Auction",
    # Reach & Frequency
    "Reach (R&F)":           "Reach & Frequency",
    "Reservation In-Feed":   "Reach & Frequency",
    # Traffic & Engagement
    "Traffic":               "Traffic & Engagement",
    "Community Interaction": "Traffic & Engagement",
    # Conversions
    "Website Conversions":   "Conversions",
    "Product Sales":         "Conversions",
    "App Promotion":         "Conversions",
    "Lead Generation":       "Conversions",
    "App Install":           "Conversions",
    "Catalog Sales":         "Conversions",
    # Other
    "Mission":               "Other",
    "Search Brand Zone":     "Other",
    "Standalone BE":         "Other",
    "Content Reservation":   "Other",
}

PRODUCT_MAP = {
    # Brand Auction
    "Brand Auction - Video View":            "Brand Auction",
    "Brand Auction - Reach":                 "Brand Auction",
    "Brand Auction - Consideration":         "Brand Auction",
    "Brand Auction - Community Interaction": "Brand Auction",
    # Reservation & Premium
    "Top Feed":                              "Reservation & Premium",
    "Openscreen":                            "Reservation & Premium",
    "Pulse Suite - Pulse Core":              "Reservation & Premium",
    "Pulse Suite - Pulse Premiere":          "Reservation & Premium",
    "Standard Feed":                         "Reservation & Premium",
    # Web & Traffic
    "Web Traffic":                           "Web & Traffic",
    "Web non-Catalog":                       "Web & Traffic",
    "Other Traffic":                         "Web & Traffic",
    # App, Commerce & Other
    "Catalog Ads":                           "App, Commerce & Other",
    "Shop Ads Custom":                       "App, Commerce & Other",
    "GMV Max":                               "App, Commerce & Other",
    "Travel Ads":                            "App, Commerce & Other",
    "App Prospecting":                       "App, Commerce & Other",
    "App Retargeting":                       "App, Commerce & Other",
    "App Traffic":                           "App, Commerce & Other",
    "App Pre-registration":                  "App, Commerce & Other",
    "Lead Form":                             "App, Commerce & Other",
    "Lead Messaging":                        "App, Commerce & Other",
    "Lead Phone Call":                       "App, Commerce & Other",
    "Content Ads - Mission":                 "App, Commerce & Other",
    "Content Ads - Search Reservation":      "App, Commerce & Other",
    "Content ads - Search Reservation":      "App, Commerce & Other",
    "Content Ads - Sponsorship":             "App, Commerce & Other",
    "Auto Ads":                              "App, Commerce & Other",
    "Streaming Ads":                         "App, Commerce & Other",
    "Others":                                "App, Commerce & Other",
}

# Sets used for upper-funnel filter
_OBJ_UPPER = {
    "Video Views (Auction)", "Reach (Auction)",
    "Reach (R&F)", "Reservation In-Feed",
}
_L3_UPPER = {
    "Brand Auction - Video View", "Brand Auction - Reach",
    "Top Feed", "Openscreen",
    "Pulse Suite - Pulse Core", "Pulse Suite - Pulse Premiere",
    "Standard Feed",
}


# ─── Default Cut Config ───────────────────────────────────────────────────────

CUTS_CONFIG = {
    "Watch Time":         {"col": "wt_group",          "df_key": "main",      "min_n": 5},
    "Creative Count":     {"col": "creative_group",    "df_key": "main",      "min_n": 5},
    "Video Duration":     {"col": "video_dur_group",   "df_key": "main",      "min_n": 5},
    "VCR":                {"col": "vcr_group",         "df_key": "main",      "min_n": 5},
    "Frequency":          {"col": "freq_group",        "df_key": "main",      "min_n": 5},
    "Weekly Impressions": {"col": "weekly_impr_group", "df_key": "main",      "min_n": 5},
    "Objective":          {"col": "obj_group",         "df_key": "objective", "min_n": 5},
    "Product Split":      {"col": "product_split",     "df_key": "product",   "min_n": 5},
    "Account Segment":    {"col": "acct_seg",          "df_key": "main",      "min_n": 5},
    "Audience Type":      {"col": "audience_type",     "df_key": "main",      "min_n": 5},
    "Billing Type":       {"col": "billing_type",      "df_key": "main",      "min_n": 5},
    "Spark Ads":          {"col": "spark_cat",         "df_key": "main",      "min_n": 5},
    "ACO":                {"col": "if_aco_bin",        "df_key": "main",      "min_n": 5},
}

QUESTION_TAGS = list(DEFAULT_QUESTION_TAGS)

# Apply per-cut enabled / min_n overrides from config.yaml
for _cut_name, _cut_yaml in _CUTS_YAML.items():
    if _cut_name in CUTS_CONFIG and isinstance(_cut_yaml, dict):
        if "enabled" in _cut_yaml:
            CUTS_CONFIG[_cut_name]["enabled"] = bool(_cut_yaml["enabled"])
        if "min_n" in _cut_yaml:
            CUTS_CONFIG[_cut_name]["min_n"]   = int(_cut_yaml["min_n"])


# ─── Domain Priors ────────────────────────────────────────────────────────────

DOMAIN_PRIORS = {
    "Watch Time": {
        "raw_col":   "paid_avg_play_dur",
        "display":   "Watch Time (s)",
        "canonical_bins": [0, 2.5, 5.0, 10.0, 100.0],
        "bin_labels":     ["<2.5s", "2.5-5s", "5-10s", ">10s"],
        "bin_rationale": {
            "<2.5s":  "Below TikTok's official Paid Video View threshold (2.5s). "
                      "Scroll-past impressions; viewer never truly engaged.",
            "2.5-5s": "TikTok's billing standard for a counted video view. "
                      "Highest-signal zone (RF importance=0.199, rank #1).",
            "5-10s":  "Extended-view territory; strong positive association with "
                      "brand-awareness and ad-recall lift.",
            ">10s":   ">10s segments represent <2% of campaigns. "
                      "Small sample; signals very high creative resonance when present.",
        },
        "min_meaningful_delta_pp": 1.5,
        "reference_sigma": 3.2,
        "forbidden_rules": [
            (
                lambda edges: not any(abs(e - 2.5) < 0.2 for e in edges),
                "No bin edge near 2.5s — TikTok's Paid Video View threshold. "
                "Without it, paid views and scroll-pasts are conflated.",
            ),
            (
                lambda edges: any(e > 20 for e in list(edges)[1:-1]),
                "A bin edge above 20s creates a near-empty tail group (<0.5% of campaigns).",
            ),
        ],
    },
    "VCR": {
        "raw_col":   "paid_vcr",
        "display":   "Video Completion Rate (%)",
        "canonical_bins": [0, 0.01, 0.02, 0.035, float("inf")],
        "bin_labels":     ["<1%", "1-2%", "2-3.5%", ">3.5%"],
        "bin_rationale": {
            "<1%":    "Very low completion; creative lost viewers within first second.",
            "1-2%":   "Below-median completion; limited brand metric movement.",
            "2-3.5%": "Mid-range completion; typical benchmark for standard TikTok creatives.",
            ">3.5%":  "High completion. RF importance=0.163 (rank #2). "
                      "Consistently stronger Brand Awareness and Intent lift.",
        },
        "min_meaningful_delta_pp": 2.0,
        "reference_sigma": 4.1,
        "forbidden_rules": [
            (
                lambda edges: any(e > 50 for e in list(edges)[:-1]),
                "VCR edges above 50% likely reflect decimal/percent unit confusion. "
                "Confirm paid_vcr is in percent form (e.g. 3.2 = 3.2%).",
            ),
        ],
    },
    "Video Duration": {
        "raw_col":   "video_dur_s",
        "display":   "Video Duration (s)",
        "canonical_bins": [0, 6, 15, 60, float("inf")],
        "bin_labels":     ["<=6s", "6-15s", "15-60s", ">60s"],
        "bin_rationale": {
            "<=6s":   "Bumper/TopView short format. High completion almost by construction.",
            "6-15s":  "Dominant TikTok in-feed format. Most reliable sub-group.",
            "15-60s": "Extended format; typically brand or product awareness campaigns.",
            ">60s":   "Long-form/native content. Sparse in BLS dataset — interpret with caution.",
        },
        "min_meaningful_delta_pp": 1.5,
        "reference_sigma": 3.8,
        "forbidden_rules": [],
    },
    "Creative Count": {
        "raw_col":   "video_material_id_count",
        "display":   "Creative Count",
        "canonical_bins": [0, 4, 9, 19, 10000],
        "bin_labels":     ["1-4", "5-9", "10-19", "20+"],
        "bin_rationale": {
            "1-4":   "Low creative diversity. Algorithm has limited room to optimise.",
            "5-9":   "Moderate rotation. RF importance=0.088 (rank #3).",
            "10-19": "Active creative testing; incremental gains over 5-9 group.",
            "20+":   "Heavy rotation. Diminishing returns territory for most verticals.",
        },
        "min_meaningful_delta_pp": 1.5,
        "reference_sigma": 3.5,
        "forbidden_rules": [
            (
                lambda edges: any(50 < e < 9998 for e in list(edges)[1:-1]),
                "A bin edge between 50 and 10,000 creates a near-empty extreme group. "
                "Consider merging into a single '20+' tail bin.",
            ),
        ],
    },
    "Frequency": {
        "raw_col":   "avg_weekly_frequency",
        "display":   "Weekly Frequency",
        "canonical_bins": None,
        "bin_labels":     ["Low", "Mid", "High"],
        "bin_rationale": {
            "Low":  "Bottom tertile (<P33). May not reach recall threshold.",
            "Mid":  "P33-P67. Standard campaign cadence; reasonable baseline.",
            "High": "Top tertile (>P67). Ad-fatigue risk; may inflate ad-recall "
                    "while showing diminishing returns on brand metric lift.",
        },
        "min_meaningful_delta_pp": 1.5,
        "reference_sigma": 3.4,
        "forbidden_rules": [],
    },
}


# ─── Preprocessing Helpers ────────────────────────────────────────────────────

def _parse_list(x):
    """Parse a string-encoded list or return a 1-element list."""
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(str(x))
    except Exception:
        return [str(x).strip()] if pd.notna(x) else []


def _compute_se(row):
    """Compute standard error from exposed/control proportions."""
    try:
        p1 = float(row["exposed_pct"])
        n1 = float(row["total_exposed_responses"])
        p2 = float(row["control_pct"])
        n2 = float(row["total_control_responses"])
        if any(pd.isna([p1, n1, p2, n2])):
            return np.nan
        if n1 <= 0 or n2 <= 0:
            return np.nan
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        return se if se > 0 else np.nan
    except Exception:
        return np.nan


def _audience_type(row):
    hc = "1" in str(row.get("is_custom_audience",  ""))
    hl = "1" in str(row.get("is_lookalike_audience", ""))
    if hc and hl:
        return "Custom+Lookalike"
    if hc:
        return "Custom"
    if hl:
        return "Lookalike"
    return "Broad"


_BILLING_MAP = {
    "GD": "GD", "CPV": "CPV", "CPM": "CPM",
    "oCPM": "Performance", "CPC": "Performance",
}

def _parse_billing(s):
    if pd.isna(s):
        return "Other"
    items   = re.findall(r'"([^"]+)"', str(s)) or [str(s).strip()]
    cleaned = [re.sub(r"[\[\]]", "", i).strip() for i in items]
    for key in ["GD", "CPV", "CPM", "oCPM", "CPC"]:
        if key in cleaned:
            return _BILLING_MAP.get(key, "Other")
    return "Other"


def _parse_flag(s):
    if pd.isna(s):
        return np.nan
    nums = re.findall(r"\d+", str(s))
    return int(nums[0]) if nums else np.nan


def _acct_seg(x):
    if pd.isna(x):
        return None
    raw = (re.findall(r'"([^"]+)"', str(x)) or [str(x)])[0]
    return raw if raw not in {"Agency", "UN_KNOWN", "Undefined"} else "Other"


# ─── Main Preprocessing Function ─────────────────────────────────────────────

def load_and_preprocess(data_path=DATA):
    """
    Load CSV, apply all cleaning/feature engineering steps, and return
    (df, df_product, df_objective) ready for BLSMetaPipeline.

    Steps
    -----
    1.  Load CSV
    2.  SE + weights (iv_weight, n_weight)
    3.  Winsorize lift (lift_w)
    4.  Feature bins: wt_group, creative_group, video_dur_group, vcr_group,
        acct_seg, audience_type, billing_type, if_aco_bin, weekly_impr_group
    5.  Frequency: freq_w, freq_group, freq_is_null
    6.  Derived columns: objective, survey_weeks, spark_cat, is_sig
    7.  Upper-funnel filter (3-signal: l1 + obj + l3)
    8.  Explode: df_product (l3_product_tag), df_objective (advertising_objective)

    Returns
    -------
    df, df_product, df_objective
    """
    print(f"Loading: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # ── 1. Weights ──────────────────────────────────────────────────────────
    df["se"]        = df.apply(_compute_se, axis=1)
    df["variance"]  = df["se"] ** 2
    df["iv_weight"] = 1 / df["variance"]
    df["n_weight"]  = (
        2 * df["total_exposed_responses"] * df["total_control_responses"]
        / (df["total_exposed_responses"] + df["total_control_responses"])
    )

    # ── 2. Winsorize lift ───────────────────────────────────────────────────
    lo = df["absolute_lift"].quantile(WINSORIZE_LO)
    hi = df["absolute_lift"].quantile(WINSORIZE_HI)
    df["lift_w"] = df["absolute_lift"].clip(lo, hi)
    print(f"  Winsorize bounds: [{lo:.4f}, {hi:.4f}]")

    # ── 3. Feature bins ─────────────────────────────────────────────────────
    df["wt_group"] = pd.cut(
        pd.to_numeric(df["paid_avg_play_dur"], errors="coerce"),
        bins=[0, 2.5, 5, 10, 100],
        labels=["<2.5s", "2.5-5s", "5-10s", ">10s"],
    )
    df["creative_group"] = pd.cut(
        pd.to_numeric(df["video_material_id_count"], errors="coerce"),
        bins=[0, 4, 9, 19, 10000],
        labels=["1-4", "5-9", "10-19", "20+"],
    )
    df["video_dur_s"] = pd.to_numeric(df["video_duration"], errors="coerce") / 10
    df["video_dur_group"] = pd.cut(
        df["video_dur_s"],
        bins=[0, 6, 15, 60, np.inf],
        labels=["<=6s", "6-15s", "15-60s", ">60s"],
    )
    df["vcr_group"] = pd.cut(
        pd.to_numeric(df["paid_vcr"], errors="coerce"),
        bins=[0, 0.01, 0.02, 0.035, np.inf],
        labels=["<1%", "1-2%", "2-3.5%", ">3.5%"],
    )
    df["acct_seg"]     = df["account_segment"].apply(_acct_seg)
    df["audience_type"] = df.apply(_audience_type, axis=1)
    df["billing_type"]  = df["bid_type"].apply(_parse_billing)
    df["if_aco_bin"]    = df["if_aco"].apply(_parse_flag)
    df["if_aeo_bin"]    = df["if_aeo"].apply(_parse_flag)
    df["is_sig"]        = pd.to_numeric(df["is_significant"], errors="coerce").fillna(0)

    # Weekly impressions
    df["weekly_impressions"] = (
        pd.to_numeric(df["reach"], errors="coerce") *
        pd.to_numeric(df["avg_weekly_frequency"], errors="coerce")
    )
    wi     = df["weekly_impressions"].dropna()
    wi_p33 = wi.quantile(0.33)
    wi_p67 = wi.quantile(0.67)
    df["weekly_impr_group"] = pd.cut(
        df["weekly_impressions"],
        bins=[0, wi_p33, wi_p67, float("inf")],
        labels=["Low", "Mid", "High"],
    )

    # ── 4. Frequency ────────────────────────────────────────────────────────
    freq_p99 = df["avg_weekly_frequency"].quantile(0.99) if FREQ_P99_CAP else np.inf
    df["freq_w"]      = pd.to_numeric(df["avg_weekly_frequency"], errors="coerce").clip(upper=freq_p99)
    df["freq_is_null"] = df["freq_w"].isna().astype(int)
    df["freq_group"]   = pd.qcut(df["freq_w"], q=3, labels=["Low", "Mid", "High"], duplicates="drop")

    # ── 5. Derived columns ──────────────────────────────────────────────────
    df["objective"]    = df["advertising_objective"]
    df["survey_weeks"] = (
        (pd.to_datetime(df["survey_end_time"]) - pd.to_datetime(df["survey_start_time"]))
        .dt.days / 7
    ).round(1)
    if "spark_cat" not in df.columns:
        df["spark_cat"] = df.get("spark_type", pd.Series(dtype=str))

    # ── 6. Upper-funnel filter (3-signal) ───────────────────────────────────
    df["l1_list"]        = df["l1_product_tag"].apply(_parse_list)
    df["obj_list_check"] = df["advertising_objective"].apply(_parse_list)
    df["l3_list_check"]  = df["l3_product_tag"].apply(_parse_list)

    df["has_upper"]     = df["l1_list"].apply(lambda ts: "Upper Funnel (Awareness)" in ts)
    df["obj_has_upper"] = df["obj_list_check"].apply(lambda ts: any(t.strip() in _OBJ_UPPER for t in ts))
    df["l3_has_upper"]  = df["l3_list_check"].apply(lambda ts: any(t.strip() in _L3_UPPER for t in ts))
    df["upper_signals"] = df["has_upper"].astype(int) + df["obj_has_upper"].astype(int) + df["l3_has_upper"].astype(int)

    before = len(df)
    df = df[df["upper_signals"] == 3].reset_index(drop=True)
    print(f"  Upper-funnel filter: {before:,} → {len(df):,} rows")

    # ── 7. Explode product & objective ──────────────────────────────────────
    df["prod_list"] = df["l3_product_tag"].apply(_parse_list)
    df["obj_list"]  = df["advertising_objective"].apply(_parse_list)

    df_product = df.explode("prod_list").copy()
    df_product = df_product[
        df_product["prod_list"].notna() &
        (df_product["prod_list"].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    df_product["product_split"] = (
        df_product["prod_list"].str.strip().map(PRODUCT_MAP).fillna("App, Commerce & Other")
    )

    df_objective = df.explode("obj_list").copy()
    df_objective = df_objective[
        df_objective["obj_list"].notna() &
        (df_objective["obj_list"].astype(str).str.strip() != "")
    ].reset_index(drop=True)
    df_objective["obj_group"] = (
        df_objective["obj_list"].str.strip().map(OBJ_MAP).fillna("Other")
    )

    print(f"  df shape: {df.shape}")
    print(f"  df_product shape: {df_product.shape} | groups: {df_product['product_split'].value_counts().to_dict()}")
    print(f"  df_objective shape: {df_objective.shape} | groups: {df_objective['obj_group'].value_counts().to_dict()}")
    return df, df_product, df_objective


# ─── Meta-Analysis Engine ─────────────────────────────────────────────────────

def _plot_meta(res_df, cut_col, label, qt_label, alpha, p_cross, test_name, cross_sig):
    """Bar chart with per-group CI and significance markers."""
    ci_lo_cols = [c for c in res_df.columns if "CI" in c and "lower" in c]
    ci_hi_cols = [c for c in res_df.columns if "CI" in c and "upper" in c]

    colors = ["#27ae60" if r == "✓" else "#95a5a6" for r in res_df["sig_vs_zero"]]
    p_str  = f"p={p_cross:.3f}" if (p_cross is not None and not np.isnan(p_cross)) else ""
    title  = f"{label} | {qt_label}\n{test_name} {p_str} {cross_sig}"

    fig, ax = plt.subplots(figsize=(max(6, len(res_df) * 1.6), 5))
    bars = ax.bar(res_df["group"], res_df["weighted_mean_lift"],
                  color=colors, alpha=0.85, width=0.6)

    if ci_lo_cols and ci_hi_cols:
        lo = res_df[ci_lo_cols[0]].values
        hi = res_df[ci_hi_cols[0]].values
        wm = res_df["weighted_mean_lift"].values
        ax.errorbar(res_df["group"], wm, yerr=[wm - lo, hi - wm],
                    fmt="none", color="black", capsize=5, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    for bar, (_, row) in zip(bars, res_df.iterrows()):
        star = " *" if row["sig_vs_zero"] == "✓" else ""
        yoff = abs(bar.get_height()) * 0.05 + 0.001
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + yoff,
                f"n={int(row['n'])}{star}",
                ha="center", va="bottom", fontsize=9)

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Weighted Mean Absolute Lift")
    ax.set_xlabel(cut_col)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    safe  = f"{label}_{qt_label}".replace(" ", "_").replace("/", "_").replace("|", "_")
    fname = os.path.join(OUT, f"{safe}.png")
    plt.savefig(fname, dpi=FIGURE_DPI)
    plt.close()
    print(f"  -> {fname}")


def meta_analyze(df_sub, cut_col, question_tags=None, label="Analysis",
                 min_n=5, weight_col="iv_weight", alpha=0.10, save_fig=SAVE_FIGS_DEFAULT,
                 mt_correction="none"):
    """
    Meta-analysis: weighted mean lift + significance per group and across groups.

    Parameters
    ----------
    df_sub        : DataFrame
    cut_col       : str    — grouping column
    question_tags : list   — question_tag values; None = all rows combined
    label         : str    — plot/print label
    min_n         : int    — minimum group size
    weight_col    : str    — 'iv_weight' or 'n_weight'
    alpha         : float  — significance level
    save_fig      : bool   — save bar charts to OUT/

    Returns
    -------
    pd.DataFrame with per-group results including:
      weighted_mean_lift, SE, CI_lower, CI_upper,
      p_vs_zero, sig_vs_zero, sig_rate_%,
      cross_group_p, cross_group_sig
    """
    run_tags = (["__ALL__"] if question_tags is None
                else list(question_tags) + ["__ALL__"])

    z_ci     = scipy_norm.ppf(1 - alpha / 2)
    all_rows = []

    for qt in run_tags:
        if qt == "__ALL__":
            df_q, qt_label = df_sub.copy(), "All Questions"
        else:
            df_q = df_sub[df_sub["question_tag"] == qt].copy()
            qt_label = qt

        df_q   = df_q.dropna(subset=[cut_col, "lift_w", weight_col])
        groups = sorted(df_q[cut_col].astype(str).unique())

        group_rows, group_arrays = [], {}

        for g in groups:
            gdf = df_q[df_q[cut_col].astype(str) == g]
            if len(gdf) < min_n:
                continue

            w  = gdf[weight_col].values.astype(float)
            y  = gdf["lift_w"].values.astype(float)
            wm = np.average(y, weights=w)

            w_sum    = w.sum()
            w_sum_sq = (w ** 2).sum()
            denom    = w_sum * (1 - w_sum_sq / w_sum ** 2) if w_sum > 0 else 0
            w_se     = (np.sqrt((w * (y - wm) ** 2).sum() / denom)
                        if (denom > 0 and len(gdf) >= 2) else np.nan)
            ci       = z_ci * w_se if not np.isnan(w_se) else np.nan

            z_stat    = wm / w_se if (not np.isnan(w_se) and w_se > 0) else np.nan
            p_vs_zero = (float(2 * (1 - scipy_norm.cdf(abs(z_stat))))
                         if not np.isnan(z_stat) else np.nan)

            ci_lo = round(wm - ci, 4) if not np.isnan(ci) else np.nan
            ci_hi = round(wm + ci, 4) if not np.isnan(ci) else np.nan

            group_rows.append({
                "question_tag":       qt_label,
                "group":              g,
                "n":                  len(gdf),
                "weighted_mean_lift": round(wm, 4),
                "SE":                 round(w_se, 4) if not np.isnan(w_se) else np.nan,
                f"CI{int((1-alpha)*100)}_lower": ci_lo,
                f"CI{int((1-alpha)*100)}_upper": ci_hi,
                "p_vs_zero":   round(p_vs_zero, 4) if not np.isnan(p_vs_zero) else np.nan,
                "sig_vs_zero": "✓" if (not np.isnan(p_vs_zero) and p_vs_zero < alpha) else "ns",
                "sig_rate_%":  round(gdf["is_sig"].mean() * 100, 1),
            })
            group_arrays[g] = y

        if not group_rows:
            continue

        arrays    = list(group_arrays.values())
        p_cross   = np.nan
        test_name = ""
        if len(arrays) == 2:
            _, p_cross = stats.ttest_ind(arrays[0], arrays[1], equal_var=False)
            test_name  = "Welch t-test"
        elif len(arrays) >= 3:
            _, p_cross = stats.f_oneway(*arrays)
            test_name  = "One-way ANOVA"

        cross_sig = "✓" if (not np.isnan(p_cross) and p_cross < alpha) else "ns"

        # ── Multiple testing correction (within cut × question_tag) ────────
        if mt_correction != "none" and len(group_rows) >= 2:
            try:
                from statsmodels.stats.multitest import multipletests
                p_raw     = np.array([r["p_vs_zero"] for r in group_rows], dtype=float)
                valid_idx = [i for i, p in enumerate(p_raw) if not np.isnan(p)]
                if len(valid_idx) >= 2:
                    method_map = {"bh": "fdr_bh", "bonferroni": "bonferroni", "holm": "holm"}
                    method     = method_map.get(mt_correction, "fdr_bh")
                    reject, q_vals, _, _ = multipletests(
                        p_raw[valid_idx], alpha=alpha, method=method)
                    for k_pos, idx in enumerate(valid_idx):
                        group_rows[idx]["p_corrected"] = round(float(q_vals[k_pos]), 4)
                        group_rows[idx]["sig_vs_zero"] = "✓" if reject[k_pos] else "ns"
                    for i in range(len(group_rows)):
                        if i not in valid_idx:
                            group_rows[i]["p_corrected"] = np.nan
            except ImportError:
                pass  # statsmodels not available

        for r in group_rows:
            r["cross_group_test"] = test_name
            r["cross_group_p"]    = round(float(p_cross), 4) if not np.isnan(p_cross) else np.nan
            r["cross_group_sig"]  = cross_sig

        all_rows.extend(group_rows)

        if len(arrays) >= 3 and not np.isnan(p_cross) and p_cross < alpha:
            tmp = df_q[df_q[cut_col].astype(str).isin(group_arrays)][["lift_w", cut_col]].dropna().copy()
            tmp[cut_col] = tmp[cut_col].astype(str)
            try:
                tukey = pairwise_tukeyhsd(tmp["lift_w"], tmp[cut_col], alpha=alpha)
                print(f"\n[{label} | {qt_label}] Tukey HSD (alpha={alpha}):")
                print(tukey.summary())
            except Exception as e:
                print(f"  Tukey HSD failed: {e}")

        print(f"\n{'='*65}")
        print(f"{label} | {qt_label}  ---  {test_name}  "
              f"{f'p={p_cross:.3f}' if not np.isnan(p_cross) else ''}  {cross_sig}")
        disp = pd.DataFrame(group_rows).set_index("group")
        disp = disp.drop(columns=["question_tag", "cross_group_test",
                                   "cross_group_p", "cross_group_sig"], errors="ignore")
        print(disp.to_string())

        if save_fig:
            _plot_meta(pd.DataFrame(group_rows), cut_col,
                       label, qt_label, alpha, p_cross, test_name, cross_sig)

    return pd.DataFrame(all_rows)


def run_all_cuts(df_main, df_prod=None, df_obj=None, question_tags=None,
                 weight_col="iv_weight", alpha=0.10):
    """Run meta_analyze() for every cut in CUTS_CONFIG."""
    if df_prod is None:
        df_prod = df_main
    if df_obj is None:
        df_obj = df_main
    if question_tags is None:
        question_tags = QUESTION_TAGS

    df_map = {"main": df_main, "product": df_prod, "objective": df_obj}

    all_results = {}
    for cut_name, cfg in CUTS_CONFIG.items():
        col    = cfg["col"]
        df_use = df_map.get(cfg["df_key"], df_main)
        if col not in df_use.columns:
            print(f"[SKIP] {cut_name}: column '{col}' not found")
            continue
        print(f"\n{'#'*70}")
        print(f"## CUT: {cut_name}  (col={col})")
        print("#" * 70)
        res = meta_analyze(
            df_use, col,
            question_tags=question_tags,
            label=cut_name,
            min_n=cfg["min_n"],
            weight_col=weight_col,
            alpha=alpha,
        )
        all_results[cut_name] = res
    return all_results


def sig_heatmap(all_results, question_tag_filter="All Questions", alpha=0.10):
    """Summary heatmap: rows = cut-group pairs, cols = question_tag."""
    rows = []
    for cut_name, res_df in all_results.items():
        if res_df is None or len(res_df) == 0:
            continue
        df_f = res_df[res_df["question_tag"] == question_tag_filter]
        if df_f.empty:
            continue
        cross_sig_flag = df_f["cross_group_sig"].iloc[0] if "cross_group_sig" in df_f.columns else "ns"
        cross_mark     = " ✓" if cross_sig_flag == "✓" else ""
        for _, row in df_f.iterrows():
            rows.append({
                "cut_group":    f"{cut_name}\n({row['group']}){cross_mark}",
                "question_tag": row["question_tag"],
                "lift":         row["weighted_mean_lift"],
                "sig_vs_zero":  row.get("sig_vs_zero", "ns"),
            })

    if not rows:
        print("No data for heatmap.")
        return

    hdf   = pd.DataFrame(rows)
    pivot = hdf.pivot_table(index="question_tag", columns="cut_group",
                             values="lift", aggfunc="mean")
    sig_p = hdf.pivot_table(index="question_tag", columns="cut_group",
                             values="sig_vs_zero",
                             aggfunc=lambda x: "✓" if "✓" in x.values else "ns")

    annot = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
    for r in pivot.index:
        for c in pivot.columns:
            try:
                val  = pivot.loc[r, c]
                star = "*" if (r in sig_p.index and c in sig_p.columns
                               and sig_p.loc[r, c] == "✓") else ""
                annot.loc[r, c] = f"{val:.3f}{star}"
            except Exception:
                pass

    # Sort columns by mean lift descending (high → low, bar-chart-like left→right)
    col_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot     = pivot[col_order]
    sig_p     = sig_p.reindex(columns=col_order)
    annot     = annot.reindex(columns=col_order)

    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 1.3),
                                     max(4, len(pivot.index) * 0.9)))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, cbar_kws={"label": "Weighted Mean Lift"})
    ax.set_title(
        f"Meta-Analysis Summary — {question_tag_filter}\n"
        f"(* = sig vs 0  |  ✓ in label = sig across groups  |  alpha={alpha})",
        fontsize=12,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Question Type")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    fname = os.path.join(OUT, f"Heatmap_{question_tag_filter.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  -> {fname}")


def _run_single_rf(df_in, num_feats, cat_feats, n_estimators, cv, top_n, label="Combined"):
    """Core RF fit; returns (imp_df, cv_r2_mean, cv_r2_std) or (None, nan, nan)."""
    ohe_dfs  = [pd.get_dummies(df_in[c].astype(str), prefix=c, drop_first=False)
                for c in cat_feats if c in df_in.columns]
    num_cols = [c for c in num_feats if c in df_in.columns]
    feat_df  = pd.concat([df_in[num_cols]] + ohe_dfs, axis=1)
    target   = df_in["lift_w"]

    mask = feat_df.notna().all(axis=1) & target.notna()
    X    = feat_df[mask].astype(float).values
    y    = target[mask].values

    if len(X) < max(cv * 4, 20):
        print(f"  [FI:{label}] Not enough samples ({len(X)}), skipping.")
        return None, np.nan, np.nan

    rf        = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="r2")
    print(f"  [FI:{label}] {cv}-fold CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  (n={len(X)})")
    rf.fit(X, y)

    imp_df = (
        pd.DataFrame({"feature": feat_df.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
    return imp_df, cv_scores.mean(), cv_scores.std()


def feature_importance(df, question_tags=None, stratify_by_tag=False,
                       n_estimators=RF_N_ESTIMATORS, cv=RF_CV_FOLDS, top_n=RF_TOP_N):
    """
    Random Forest feature importance.

    Parameters
    ----------
    stratify_by_tag : bool
        If True and question_tags has ≥2 entries, runs a separate RF per tag
        and generates a side-by-side comparison plot.
    """
    num_feats           = ["paid_avg_play_dur", "video_material_id_count", "paid_vcr", "freq_is_null"]
    cat_feats_combined  = ["objective", "wt_group", "creative_group", "survey_region",
                            "question_tag", "spark_cat", "audience_type"]
    cat_feats_per_tag   = ["objective", "wt_group", "creative_group", "survey_region",
                            "spark_cat", "audience_type"]  # exclude question_tag when stratifying

    # Decide whether to stratify
    do_stratify = (
        stratify_by_tag
        and question_tags
        and "question_tag" in df.columns
        and len([qt for qt in question_tags if qt in df["question_tag"].unique()]) >= 2
    )

    if do_stratify:
        tags_present = [qt for qt in question_tags if qt in df["question_tag"].unique()]
        fi_results   = {}
        for qt in tags_present:
            df_q = df[df["question_tag"] == qt].copy()
            imp_df, r2_mean, _ = _run_single_rf(
                df_q, num_feats, cat_feats_per_tag, n_estimators, cv, top_n, label=qt)
            if imp_df is not None:
                fi_results[qt] = imp_df

        if not fi_results:
            print("[FI] No valid per-tag results.")
            return None

        # ── Multi-panel comparison plot ──────────────────────────────────────
        n_tags  = len(fi_results)
        palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
        fig, axes = plt.subplots(1, n_tags,
                                  figsize=(8 * n_tags, max(6, top_n * 0.45 + 1)))
        if n_tags == 1:
            axes = [axes]

        x_max = max(imp_df["importance"].max()
                    for imp_df in fi_results.values()) * 1.15

        for ax, (qt, imp_df), color in zip(axes, fi_results.items(), palette):
            ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
                    color=color, alpha=0.82)
            ax.set_title(f"{qt}", fontsize=11)
            ax.set_xlabel("Importance")
            ax.set_xlim(0, max(x_max, 0.05))

        plt.suptitle(f"Feature Importance by Question Tag — Top {top_n}",
                     fontsize=13, y=1.01)
        plt.tight_layout()
        fname = os.path.join(OUT, "Pipeline_FI_by_Tag.png")
        plt.savefig(fname, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        print(f"  -> {fname}")
        for qt, imp_df in fi_results.items():
            print(f"\n[FI: {qt}]")
            print(imp_df.to_string(index=False))
        return fi_results

    else:
        # ── Combined mode (original behaviour) ──────────────────────────────
        imp_df, r2_mean, r2_std = _run_single_rf(
            df, num_feats, cat_feats_combined, n_estimators, cv, top_n, label="Combined")
        if imp_df is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1],
                color="#3498db", alpha=0.82)
        ax.set_title(
            f"Top {top_n} Feature Importances (RF)  —  CV R²={r2_mean:.3f}", fontsize=11)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        fname = os.path.join(OUT, "Pipeline_Feature_Importance.png")
        plt.savefig(fname, dpi=FIGURE_DPI)
        plt.close()
        print(f"  -> {fname}")
        print(imp_df.to_string(index=False))
        return imp_df


# ─── BLSMetaPipeline Class ────────────────────────────────────────────────────

class BLSMetaPipeline:
    """
    Chainable OO wrapper for BLS meta-analysis.

    Quick start
    -----------
    from bls_meta_pipeline import BLSMetaPipeline, load_and_preprocess

    df, df_product, df_objective = load_and_preprocess("path/to/data.csv")
    pipeline = BLSMetaPipeline(df, df_product=df_product, df_objective=df_objective)
    pipeline.full_run()

    # Or step by step:
    pipeline.configure(alpha=0.05)
    pipeline.add_cut("Region", col="survey_region")
    results = pipeline.run_all()
    pipeline.all_heatmaps()
    pipeline.run_cross_cut("Watch Time", "VCR")
    pipeline.cross_heatmap("Watch Time", "VCR", question_tag="AD_RECALL")
    pipeline.feature_importance()
    pipeline.recommend(top_n=3)
    pipeline.export()
    """

    _DEFAULT_CONFIG = dict(
        alpha=DEFAULT_ALPHA,
        weight_col=DEFAULT_WEIGHT_COL,
        min_n=DEFAULT_MIN_N,
        question_tags=None,
        mt_correction="none",
        fi_stratify_by_tag=False,
        winsorize_lower=0.02,
        winsorize_upper=0.98,
    )

    def __init__(self, df_main, df_product=None, df_objective=None, config=None):
        self._df      = df_main.copy()
        self._df_prod = df_product.copy()  if df_product  is not None else df_main.copy()
        self._df_obj  = df_objective.copy() if df_objective is not None else df_main.copy()
        self._cuts    = dict(CUTS_CONFIG)
        self._cfg     = dict(self._DEFAULT_CONFIG)
        self._cfg["question_tags"] = list(QUESTION_TAGS)
        if config:
            self._cfg.update(config)
        self._results          = {}
        self._cross_results    = {}
        self._fi_result        = None    # stored after feature_importance()
        self._last_report_path = None    # stored after generate_report()
        # Runtime flags — overrideable by launch_ui() / _apply_config()
        self._fi_enabled      = FI_ENABLED
        self._cc_enabled      = CC_ENABLED
        self._cc_pairs        = [list(p) for p in CC_DEFAULT_PAIRS]
        self._out_cfg         = dict(_OUT_YAML)
        self._report_cfg      = {"enabled": True, "ai_api_key": "", "ai_model": "gpt-4o-mini"}
        self._advertiser_cfg  = {}   # advertiser ID filter (enabled, ids, id_column)

    # ── Configuration ─────────────────────────────────────────────────────────
    def configure(self, **kwargs):
        """Update global config: alpha, weight_col, min_n, question_tags."""
        self._cfg.update(kwargs)
        return self

    def _apply_config(self, cfg: dict):
        """
        Apply a full config dict (from launch_ui or a yaml load) to this
        pipeline instance.  Only keys present in cfg are updated.
        """
        ana = cfg.get("analysis", {})
        if "alpha"          in ana: self._cfg["alpha"]          = float(ana["alpha"])
        if "weight_col"     in ana: self._cfg["weight_col"]     = str(ana["weight_col"])
        if "min_n"          in ana: self._cfg["min_n"]          = int(ana["min_n"])
        if "question_tags"  in ana: self._cfg["question_tags"]  = list(ana["question_tags"])
        if "mt_correction"  in ana: self._cfg["mt_correction"]  = str(ana["mt_correction"])
        pre = cfg.get("preprocessing", {})
        if "winsorize_lower" in pre: self._cfg["winsorize_lower"] = float(pre["winsorize_lower"])
        if "winsorize_upper" in pre: self._cfg["winsorize_upper"] = float(pre["winsorize_upper"])
        if "freq_p99_cap"    in pre: self._cfg["freq_p99_cap"]    = bool(pre["freq_p99_cap"])

        for name, cut_cfg in cfg.get("cuts", {}).items():
            if name in self._cuts and isinstance(cut_cfg, dict):
                if "enabled" in cut_cfg:
                    self._cuts[name]["enabled"] = bool(cut_cfg["enabled"])
                if "min_n" in cut_cfg:
                    self._cuts[name]["min_n"]   = int(cut_cfg["min_n"])

        cc = cfg.get("cross_cuts", {})
        if "enabled"          in cc: self._cc_enabled           = bool(cc["enabled"])
        if "pairs"            in cc: self._cc_pairs             = [list(p) for p in cc["pairs"]]
        if "min_n_multiplier" in cc: self._cfg["cc_min_n_mult"] = int(cc["min_n_multiplier"])

        fi = cfg.get("feature_importance", {})
        if "enabled"         in fi: self._fi_enabled                = bool(fi["enabled"])
        if "n_estimators"    in fi: self._cfg["fi_n_estimators"]    = int(fi["n_estimators"])
        if "cv_folds"        in fi: self._cfg["fi_cv_folds"]        = int(fi["cv_folds"])
        if "top_n"           in fi: self._cfg["fi_top_n"]           = int(fi["top_n"])
        if "stratify_by_tag" in fi: self._cfg["fi_stratify_by_tag"] = bool(fi["stratify_by_tag"])

        rpt = cfg.get("report", {})
        if "enabled"     in rpt: self._report_cfg["enabled"]     = bool(rpt["enabled"])
        if "ai_api_key"  in rpt: self._report_cfg["ai_api_key"]  = str(rpt["ai_api_key"])
        if "ai_model"    in rpt: self._report_cfg["ai_model"]    = str(rpt["ai_model"])

        out = cfg.get("output", {})
        self._out_cfg = {**self._out_cfg, **out}

        adv = cfg.get("advertiser_filter", {})
        if adv:
            self._advertiser_cfg = dict(adv)

        for ccut in cfg.get("custom_cuts", []):
            try:
                self._register_custom_cut(ccut)
            except Exception as e:
                print(f"  [CustomCut] Failed to register '{ccut.get('name','')}': {e}")
        return self

    def _register_custom_cut(self, ccut_cfg: dict):
        """
        Dynamically register a user-defined cut from a config dict.

        Supported bin_type values
        -------------------------
        categorical : use the column values as-is (astype str)
        quantile    : pd.qcut(col, q=N, labels=[...])
        fixed       : pd.cut(col, bins=[...], labels=[...])
        """
        name     = ccut_cfg.get("name", "").strip()
        col      = ccut_cfg.get("col",  "").strip()
        if not name or not col:
            return
        if name in self._cuts:          # already registered (e.g. re-apply config)
            return

        bin_type = ccut_cfg.get("bin_type", "categorical")
        df_key   = ccut_cfg.get("df_key",  "main")
        min_n    = int(ccut_cfg.get("min_n", self._cfg["min_n"]))
        enabled  = bool(ccut_cfg.get("enabled", True))

        if bin_type == "categorical":
            fn = lambda df, c=col: df[c].astype(str).where(df[c].notna(), other=None)
            self.add_cut(name, fn=fn, df_key=df_key, min_n=min_n)

        elif bin_type == "quantile":
            q      = int(ccut_cfg.get("q", 3))
            labels = ccut_cfg.get("labels") or None
            fn = lambda df, c=col, qq=q, lb=labels: pd.qcut(
                pd.to_numeric(df[c], errors="coerce"),
                q=qq, labels=lb, duplicates="drop")
            self.add_cut(name, fn=fn, df_key=df_key, min_n=min_n)

        elif bin_type == "fixed":
            bins   = ccut_cfg.get("bins", [])
            labels = ccut_cfg.get("labels") or None
            self.add_cut(name, col=col, bins=bins, labels=labels,
                         df_key=df_key, min_n=min_n)

        if name in self._cuts:
            self._cuts[name]["enabled"] = enabled

    def _apply_advertiser_filter(self):
        """
        Filter internal DataFrames to only rows matching the configured
        advertiser IDs.  No-op when disabled or IDs list is empty.
        Call at the start of full_run() so every analysis uses the same subset.
        """
        adv = self._advertiser_cfg
        if not adv.get("enabled", False):
            return
        ids = [str(i).strip() for i in adv.get("ids", []) if str(i).strip()]
        if not ids:
            return
        id_col = adv.get("id_column", "advertiser_id")
        for attr in ("_df", "_df_prod", "_df_obj"):
            df = getattr(self, attr)
            if id_col in df.columns:
                setattr(self, attr,
                        df[df[id_col].astype(str).isin(ids)].reset_index(drop=True))
        print(f"  [Filter] Advertiser filter: {len(ids)} ID(s) → {len(self._df):,} rows remaining")

    def set_metrics(self, lst):
        """Replace question_tags list."""
        self._cfg["question_tags"] = list(lst)
        return self

    def add_metric(self, question_tag):
        """Append a single question_tag."""
        if question_tag not in self._cfg["question_tags"]:
            self._cfg["question_tags"].append(question_tag)
        return self

    # ── Cut management ────────────────────────────────────────────────────────
    def add_cut(self, name, col=None, df_key="main", min_n=None,
                bins=None, labels=None, fn=None, where=None,
                alpha=None, question_tags=None):
        """Register a new cut dimension (chainable)."""
        binned_col = col
        if fn is not None:
            slug   = name.lower().replace(" ", "_").replace("-", "_")
            derived = f"_fn_{slug}"
            for _df in [self._df, self._df_prod, self._df_obj]:
                _df[derived] = fn(_df)
            binned_col = derived
        elif bins is not None:
            target = f"{col}_grp"
            for _df in [self._df, self._df_prod, self._df_obj]:
                if col in _df.columns:
                    _df[target] = pd.cut(pd.to_numeric(_df[col], errors="coerce"),
                                         bins=bins, labels=labels)
            binned_col = target

        if binned_col is None:
            raise ValueError(f"add_cut('{name}'): supply col=, bins=, or fn=.")

        self._cuts[name] = {
            "col":            binned_col,
            "raw_col":        col if bins is not None else None,
            "bins":           bins,
            "df_key":         df_key,
            "min_n":          min_n if min_n is not None else self._cfg["min_n"],
            "where_fn":       where,
            "alpha_override": alpha,
            "qt_override":    question_tags,
        }
        return self

    def remove_cut(self, name):
        self._cuts.pop(name, None)
        return self

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _df_for_key(self, key):
        if key == "product":   return self._df_prod
        if key == "objective": return self._df_obj
        return self._df

    def _run_meta(self, name):
        cfg    = self._cuts[name]
        col    = cfg["col"]
        df_use = self._df_for_key(cfg.get("df_key", "main"))
        if cfg.get("where_fn"):
            df_use = df_use[cfg["where_fn"](df_use)]
        if col not in df_use.columns:
            print(f"[SKIP] {name}: column '{col}' not found")
            return None
        qt    = cfg.get("qt_override")    or self._cfg["question_tags"]
        alpha = cfg.get("alpha_override") or self._cfg["alpha"]
        return meta_analyze(df_use, col,
                            question_tags=qt, label=name,
                            min_n=cfg["min_n"],
                            weight_col=self._cfg["weight_col"],
                            alpha=alpha,
                            save_fig=self._out_cfg.get("save_figures", SAVE_FIGS_DEFAULT),
                            mt_correction=self._cfg.get("mt_correction", "none"))

    # ── Audit ─────────────────────────────────────────────────────────────────
    def audit_cut(self, name, verbose=True):
        """7-layer domain + statistical audit for one cut."""
        from statsmodels.stats.power import TTestIndPower
        from scipy import stats as _stats

        if name not in self._cuts:
            raise KeyError(f"Cut '{name}' not registered.")

        cfg  = self._cuts[name]
        col  = cfg["col"]
        df   = self._df_for_key(cfg.get("df_key", "main"))
        if cfg.get("where_fn"):
            try: df = df[cfg["where_fn"](df)]
            except Exception: pass

        if col not in df.columns:
            return False, {"ERROR": [f"Column '{col}' not found."]}

        series   = df[col].dropna()
        counts   = series.value_counts().sort_index()
        k        = len(counts)
        n_total  = len(series)
        n_all    = len(df[col])
        coverage = n_total / n_all if n_all > 0 else 0

        errors, warnings, info = [], [], []
        prior = DOMAIN_PRIORS.get(name, {})
        alpha = cfg.get("alpha_override") or self._cfg["alpha"]

        # 1. Coverage
        if coverage < 0.90:
            errors.append(f"Coverage {coverage*100:.1f}% — over 10% excluded.")
        elif coverage < 0.97:
            warnings.append(f"Coverage {coverage*100:.1f}%.")
        else:
            info.append(f"Coverage {coverage*100:.1f}% — OK.")

        # 2. Min group size
        if counts.min() < cfg.get("min_n", self._cfg["min_n"]):
            errors.append(f"Group '{counts.idxmin()}' has {counts.min()} campaigns — below min_n.")

        # 3. Statistical power
        delta   = prior.get("min_meaningful_delta_pp", 2.0)
        sigma   = prior.get("reference_sigma") or (df["lift_w"].std() if "lift_w" in df.columns else 3.5)
        n_harm  = k / sum(1 / max(n, 1) for n in counts)
        d       = delta / sigma
        pwr_now = TTestIndPower().power(effect_size=d, nobs1=n_harm, alpha=alpha, ratio=1.0)
        if   pwr_now < 0.50: errors.append(f"Insufficient power ({pwr_now*100:.0f}%).")
        elif pwr_now < 0.70: warnings.append(f"Marginal power ({pwr_now*100:.0f}%).")
        else:                info.append(f"Power {pwr_now*100:.0f}% — OK.")

        # 4. Imbalance
        imbalance = counts.max() / max(counts.min(), 1)
        if imbalance > 20:   warnings.append(f"Severe imbalance {imbalance:.0f}x.")
        elif imbalance > 5:  info.append(f"Moderate imbalance ({imbalance:.1f}x).")

        # 5. Multiple comparisons
        n_pairs = k * (k - 1) // 2
        if k > 8:   errors.append(f"{k} groups — {n_pairs} pairs. Reduce to ≤8.")
        elif k > 5: warnings.append(f"{k} groups — {n_pairs} pairs.")

        # 6. Levene
        if "lift_w" in df.columns:
            grp_data = [df.loc[df[col] == g, "lift_w"].dropna().values
                        for g in counts.index if (df[col] == g).sum() >= 2]
            if len(grp_data) >= 2:
                _, p_lev = _stats.levene(*grp_data)
                if p_lev < 0.05:
                    warnings.append(f"Levene p={p_lev:.3f} — heterogeneous variance.")
                else:
                    info.append(f"Levene p={p_lev:.3f} — OK.")

        # 7. Domain forbidden rules (custom bins only)
        edges = cfg.get("bins")
        if edges is not None:
            for rule_fn, reason in prior.get("forbidden_rules", []):
                try:
                    if rule_fn(edges):
                        errors.append(f"[Domain] {reason}")
                except Exception:
                    pass

        report = {}
        if errors:   report["ERROR"]          = errors
        if warnings: report["WARNING"]        = warnings
        if info:     report["INFO"]           = info

        is_valid = len(errors) == 0

        if verbose:
            status = "PASS" if is_valid else "FAIL"
            print(f"\n{'='*60}")
            print(f"  Audit: {name}  [{status}]")
            print(f"  Groups: {dict(counts)}")
            for level, icon in [("ERROR","✗"), ("WARNING","⚠"), ("INFO","ℹ")]:
                for msg in report.get(level, []):
                    print(f"  {icon} {msg}")
            print("=" * 60)

        return is_valid, report

    def audit_all(self, verbose=True):
        """Audit all registered cuts."""
        summary = {}
        for name in self._cuts:
            valid, rep = self.audit_cut(name, verbose=verbose)
            summary[name] = {"valid": valid,
                             "n_errors": len(rep.get("ERROR", [])),
                             "n_warnings": len(rep.get("WARNING", []))}
        passed = sum(1 for v in summary.values() if v["valid"])
        print(f"\n{'─'*50}")
        print(f"  Audit: {passed}/{len(summary)} cuts passed.")
        failed = [n for n, v in summary.items() if not v["valid"]]
        if failed:
            print(f"  Failed: {failed}")
        print("─" * 50)
        return summary

    # ── Single / All cuts ─────────────────────────────────────────────────────
    def run_cut(self, name, strict=False):
        """Run a single cut (with automatic pre-flight audit)."""
        valid, rep = self.audit_cut(name, verbose=False)
        if not valid:
            self.audit_cut(name, verbose=True)
            if strict:
                print(f"  [BLOCKED] '{name}' failed audit.")
                return None
            print(f"  [WARNING] Running '{name}' despite audit failures.")
        elif rep.get("WARNING"):
            for msg in rep["WARNING"]:
                print(f"  ⚠ [{name}] {msg}")
        res = self._run_meta(name)
        if res is not None:
            self._results[name] = res
        return res

    def run_all(self, cuts=None, strict=False):
        """Run all (or subset of) registered cuts.  Disabled cuts are skipped."""
        for name in (cuts or list(self._cuts.keys())):
            if not self._cuts.get(name, {}).get("enabled", True):
                print(f"[SKIP] '{name}' is disabled in config.")
                continue
            self.run_cut(name, strict=strict)
        return self._results

    # ── Cross-cut analysis ────────────────────────────────────────────────────
    def run_cross_cut(self, *cut_names, question_tags=None, min_n=None, df_key="main"):
        """Multi-dimensional cross analysis (2+ cuts)."""
        if len(cut_names) < 2:
            raise ValueError("run_cross_cut() requires at least 2 cut names.")

        cols = []
        for n in cut_names:
            if n not in self._cuts:
                raise KeyError(f"Cut '{n}' not registered.")
            cols.append(self._cuts[n]["col"])

        df = self._df_for_key(df_key).copy()
        for n in cut_names:
            fn = self._cuts[n].get("where_fn")
            if fn:
                try: df = df[fn(df)]
                except Exception: pass

        qt_list  = question_tags or self._cfg["question_tags"]
        mn       = min_n if min_n is not None else max(self._cfg["min_n"] * 3, 10)
        alpha    = self._cfg["alpha"]
        wcol     = self._cfg["weight_col"]

        rows = []
        for qtag in qt_list:
            sub = df[df["question_tag"] == qtag].copy() if "question_tag" in df.columns else df.copy()
            sub = sub.dropna(subset=cols + ["lift_w"])
            for keys, group in sub.groupby(cols, observed=True):
                keys_t = keys if isinstance(keys, tuple) else (keys,)
                n      = len(group)
                if n < mn:
                    continue
                w    = group[wcol].fillna(0).clip(lower=0) if wcol in group.columns else pd.Series(np.ones(n))
                wsum = w.sum()
                wml  = (group["lift_w"] * w).sum() / wsum if wsum > 0 else group["lift_w"].mean()
                se   = group["lift_w"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
                if n >= 3:
                    t_stat, p_val = stats.ttest_1samp(group["lift_w"].dropna(), 0)
                else:
                    t_stat, p_val = np.nan, np.nan
                row = dict(zip(cut_names, keys_t))
                row.update({
                    "question_tag":       qtag,
                    "n":                  n,
                    "weighted_mean_lift": round(float(wml),  4) if pd.notna(wml)   else np.nan,
                    "se":                 round(float(se),   4) if pd.notna(se)    else np.nan,
                    "p_vs_zero":          round(float(p_val),4) if pd.notna(p_val) else np.nan,
                    "sig_vs_zero":        "✓" if pd.notna(p_val) and p_val < alpha else "✗",
                })
                rows.append(row)

        result_df = pd.DataFrame(rows) if rows else pd.DataFrame()
        key = " × ".join(cut_names)
        self._cross_results[key] = result_df

        if result_df.empty:
            print(f"[WARNING] run_cross_cut({key}): no cells passed min_n={mn}.")
        else:
            n_sig = (result_df["sig_vs_zero"] == "✓").sum()
            print(f"run_cross_cut({key}): {len(result_df)} cells, {n_sig} significant at α={alpha}.")
        return result_df

    def cross_heatmap(self, cut1, cut2, question_tag=None,
                      metric="weighted_mean_lift", min_n=None,
                      figsize=None, show_n=True, annotate_sig=True):
        """2D heatmap of cross-cut lift."""
        key = f"{cut1} × {cut2}"
        if key not in self._cross_results:
            self.run_cross_cut(cut1, cut2, min_n=min_n)

        df = self._cross_results[key]
        if df.empty:
            print(f"No data for {key}.")
            return None

        qt = question_tag or (self._cfg["question_tags"][0] if self._cfg["question_tags"] else None)
        if qt and "question_tag" in df.columns:
            df = df[df["question_tag"] == qt]
        if df.empty:
            print(f"No data for {key} | {qt}.")
            return None

        pivot   = df.pivot_table(index=cut1, columns=cut2, values=metric, aggfunc="mean")
        n_pivot = df.pivot_table(index=cut1, columns=cut2, values="n",    aggfunc="sum")
        s_pivot = df.pivot_table(index=cut1, columns=cut2, values="sig_vs_zero",
                                 aggfunc=lambda x: "✓" if (x == "✓").any() else "✗")

        annot = pivot.copy().astype(object)
        for r in pivot.index:
            for c in pivot.columns:
                try:
                    v = pivot.loc[r, c]
                    n = n_pivot.loc[r, c] if (r in n_pivot.index and c in n_pivot.columns) else np.nan
                    s = s_pivot.loc[r, c] if (r in s_pivot.index and c in s_pivot.columns) else ""
                    if pd.isna(v):
                        annot.loc[r, c] = "—"
                    else:
                        star = "*" if annotate_sig and s == "✓" else ""
                        txt  = f"{v:.2f}{star}"
                        if show_n and pd.notna(n):
                            txt += f"\n(n={int(n)})"
                        annot.loc[r, c] = txt
                except Exception:
                    annot.loc[r, c] = ""

        if figsize is None:
            figsize = (max(6, len(pivot.columns) * 2.0), max(4, len(pivot.index) * 1.6))

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot.astype(float), annot=annot, fmt="",
                    cmap="RdYlGn", center=0, linewidths=0.5,
                    cbar_kws={"label": "Weighted Mean Lift (pp)"}, ax=ax)
        ax.set_title(f"{cut1}  ×  {cut2}   [{qt}]", fontsize=11, pad=12)
        ax.set_xlabel(cut2)
        ax.set_ylabel(cut1)
        plt.tight_layout()
        safe = lambda s: s.replace(" ", "_").replace("/", "-")
        fname = os.path.join(OUT, f"cross__{safe(cut1)}_x_{safe(cut2)}__{qt}.png")
        plt.savefig(fname, dpi=FIGURE_DPI)
        plt.close()
        print(f"* = significant vs zero at α={self._cfg['alpha']}  |  saved → {fname}")
        return pivot

    def all_cross_heatmaps(self, cut_pairs=None, question_tag=None, **kwargs):
        """Run cross_heatmap for multiple cut pairs."""
        from itertools import combinations
        if cut_pairs is None:
            top5 = list(self._cuts.keys())[:5]
            cut_pairs = list(combinations(top5, 2))
        for c1, c2 in cut_pairs:
            try:
                self.cross_heatmap(c1, c2, question_tag=question_tag, **kwargs)
            except Exception as e:
                print(f"[SKIP] cross_heatmap({c1}, {c2}): {e}")

    # ── Visualisation ─────────────────────────────────────────────────────────
    def heatmap(self, question_tag="All Questions"):
        sig_heatmap(self._results, question_tag_filter=question_tag,
                    alpha=self._cfg["alpha"])

    def all_heatmaps(self):
        for qt in self._cfg["question_tags"] + ["All Questions"]:
            self.heatmap(question_tag=qt)

    # ── Feature importance ────────────────────────────────────────────────────
    def feature_importance(self, n_estimators=None, cv=None, top_n=None,
                            stratify_by_tag=None):
        if not self._fi_enabled:
            print("[SKIP] Feature importance is disabled "
                  "(config: feature_importance.enabled = false).")
            return None
        result = feature_importance(
            self._df,
            question_tags=self._cfg.get("question_tags"),
            stratify_by_tag=(stratify_by_tag
                             if stratify_by_tag is not None
                             else self._cfg.get("fi_stratify_by_tag", False)),
            n_estimators=n_estimators or self._cfg.get("fi_n_estimators", RF_N_ESTIMATORS),
            cv=cv         or self._cfg.get("fi_cv_folds",     RF_CV_FOLDS),
            top_n=top_n   or self._cfg.get("fi_top_n",        RF_TOP_N),
        )
        self._fi_result = result
        return result

    # ── Recommendations ───────────────────────────────────────────────────────
    def recommend(self, top_n=3):
        """Rank cut-group combos by significance + lift."""
        if not self._results:
            print("No results yet — run run_all() first.")
            return None
        combined = pd.concat(
            [r.assign(cut=k) for k, r in self._results.items()
             if r is not None and len(r) > 0],
            ignore_index=True,
        )
        combined["sig_score"]  = (combined["sig_vs_zero"] == "✓").astype(int)
        combined["rank_score"] = combined["sig_score"] + combined["weighted_mean_lift"]
        top = (combined.sort_values("rank_score", ascending=False)
                        .groupby("question_tag").head(top_n)
                        .reset_index(drop=True))
        print(f"\nTop {top_n} recommendations per question_tag:")
        print(top[["question_tag", "cut", "group", "weighted_mean_lift",
                   "sig_vs_zero", "n"]].to_string(index=False))
        return top

    # ── Export ────────────────────────────────────────────────────────────────
    def export(self, path=None, fmt="csv", include_cross=False):
        """Export all results to disk."""
        parts = [r.assign(cut=k) for k, r in self._results.items()
                 if r is not None and len(r) > 0]
        if include_cross:
            for k, r in self._cross_results.items():
                if r is not None and len(r) > 0:
                    parts.append(r.assign(cut=k))
        if not parts:
            print("Nothing to export.")
            return
        combined = pd.concat(parts, ignore_index=True)
        if path is None:
            suffix = "_with_cross" if include_cross else ""
            path = os.path.join(OUT, f"meta_analysis_results{suffix}.{'xlsx' if fmt == 'excel' else 'csv'}")
        if fmt == "excel":
            combined.to_excel(path, index=False)
        else:
            combined.to_csv(path, index=False)
        print(f"Exported {len(combined):,} rows → {path}")
        return combined

    # ── Interactive Config UI ─────────────────────────────────────────────────
    def launch_ui(self):
        """
        Open the interactive Config UI to adjust all pipeline parameters.

        - Jupyter / Google Colab → ipywidgets inline panel  (non-blocking)
        - Local script          → tkinter popup window      (blocking)

        Clicking **Apply & Run** applies settings to this instance
        and calls full_run() automatically.

        Example (Colab cell)::

            pipeline.launch_ui()
        """
        import sys, os as _os
        _dir = _os.path.dirname(_os.path.abspath(_CONFIG_PATH))
        if _dir not in sys.path:
            sys.path.insert(0, _dir)
        try:
            from config_ui import launch_ui as _launch_ui  # noqa
        except ImportError:
            print("[ERROR] config_ui.py not found next to bls_meta_pipeline.py")
            return self

        def _on_run(updated_cfg):
            self._apply_config(updated_cfg)
            print("\n🚀  Pipeline running with updated config …")
            self.full_run()

        _launch_ui(config_path=_CONFIG_PATH, on_run=_on_run)
        return self

    # ── Report generation ─────────────────────────────────────────────────────
    def generate_report(self, ai_api_key: str = "", ai_model: str = "") -> str:
        """
        Generate a self-contained HTML report in OUT directory.
        Returns path to the HTML file.
        """
        try:
            from report_generator import generate_report as _gen_report
        except ImportError:
            import sys, os as _os
            _dir = _os.path.dirname(_os.path.abspath(__file__))
            if _dir not in sys.path:
                sys.path.insert(0, _dir)
            from report_generator import generate_report as _gen_report

        key   = ai_api_key or self._report_cfg.get("ai_api_key", "")
        model = ai_model   or self._report_cfg.get("ai_model",   "gpt-4o-mini")
        n_campaigns = len(self._df)

        return _gen_report(
            output_dir  = OUT,
            results     = self._results,
            config      = self._cfg,
            fi_result   = self._fi_result,
            n_campaigns = n_campaigns,
            ai_api_key  = key,
            ai_model    = model,
        )

    # ── One-shot full run ──────────────────────────────────────────────────────
    def full_run(self, strict=False, cross_pairs=None):
        """
        Complete pipeline in one call:
        run_all → all_heatmaps → all_cross_heatmaps → feature_importance → recommend → export
        Respects per-instance flags set by _apply_config() / launch_ui().
        """
        # Apply advertiser filter before any analysis
        self._apply_advertiser_filter()

        self.run_all(strict=strict)

        # Clean up stale PNGs from previous runs so report only shows current-run charts
        import glob as _glob
        for _pattern in ("Heatmap_*.png", "Pipeline_F*.png", "cross_*.png"):
            for _old in _glob.glob(os.path.join(OUT, _pattern)):
                try:
                    os.remove(_old)
                except Exception:
                    pass

        self.all_heatmaps()
        _pairs = cross_pairs or (self._cc_pairs if self._cc_enabled else [])
        if _pairs:
            self.all_cross_heatmaps(_pairs)
        self.feature_importance()   # stores to self._fi_result
        self.recommend()
        if self._out_cfg.get("export_excel", EXPORT_EXCEL_DEFAULT):
            self.export(fmt="excel")
        elif self._out_cfg.get("export_csv", EXPORT_CSV_DEFAULT):
            self.export(fmt="csv")
        if self._report_cfg.get("enabled", True):
            try:
                self._last_report_path = self.generate_report()
            except Exception as e:
                print(f"  [Report] Generation failed: {e}")
                self._last_report_path = None


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("BLS Meta-Analysis Pipeline")
    print("=" * 70)

    df, df_product, df_objective = load_and_preprocess(DATA)

    pipeline = BLSMetaPipeline(
        df,
        df_product=df_product,
        df_objective=df_objective,
    )

    pipeline.full_run(
        cross_pairs=[("Watch Time", "VCR"), ("Watch Time", "Objective")]
    )
