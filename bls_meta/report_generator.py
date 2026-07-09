"""
BLS Meta-Analysis — HTML Report Generator
==========================================
Generates a self-contained HTML report (all charts embedded as base64).

Usage
-----
    from report_generator import generate_report

    report_path = generate_report(
        output_dir  = "output/",
        results     = pipeline._results,
        config      = {"alpha": 0.10, "weight_col": "iv_weight", ...},
        fi_result   = pipeline._fi_result,     # DataFrame or dict or None
        n_campaigns = 50000,
        ai_api_key  = "sk-...",                # optional; empty = use built-in key
        ai_model    = "gemini-2.0-flash",      # optional
    )
    # → writes output/report_YYYYMMDD_HHMMSS.html and returns its path
"""
from __future__ import annotations
import os, base64, datetime
from typing import Optional

# ── Built-in default AI key ───────────────────────────────────────────────
# Gemini Flash (free tier) — replace with your key from aistudio.google.com
_BUILTIN_AI_KEY   = ""  # set your Gemini key here, or leave empty to disable built-in
_BUILTIN_AI_MODEL = "gemini-2.0-flash"

import numpy  as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════════
_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#f0f2f5;color:#1a1a2e;line-height:1.6}
.container{max-width:1280px;margin:0 auto;padding:24px}
/* ── header ── */
.header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        color:white;border-radius:14px;padding:36px;margin-bottom:24px}
.header h1{font-size:2rem;margin-bottom:8px;letter-spacing:-.5px}
.header .meta{opacity:.75;font-size:.88rem;margin-top:4px}
.header .meta span{margin-right:16px}
/* ── card ── */
.card{background:white;border-radius:12px;padding:24px;margin-bottom:20px;
      box-shadow:0 2px 12px rgba(0,0,0,.07)}
.card h2{font-size:1.2rem;color:#1a1a2e;margin-bottom:16px;padding-bottom:10px;
         border-bottom:2px solid #f0f2f5;display:flex;align-items:center;gap:8px}
.card h3{font-size:1rem;color:#444;margin:20px 0 8px}
/* ── KPI row ── */
.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:4px}
.kpi-card{background:#f8f9fa;border-radius:10px;padding:18px;text-align:center}
.kpi-value{font-size:2rem;font-weight:700;color:#1a1a2e}
.kpi-label{font-size:.8rem;color:#777;margin-top:4px;text-transform:uppercase;
           letter-spacing:.5px}
/* ── AI box ── */
.ai-box{background:linear-gradient(135deg,#667eea18,#764ba218);
        border:1px solid #667eea35;border-radius:10px;padding:22px;
        line-height:1.9;font-size:.95rem}
.ai-box p{margin-bottom:14px}.ai-box p:last-child{margin-bottom:0}
/* ── table ── */
table{width:100%;border-collapse:collapse;font-size:.88rem}
th{background:#f8f9fa;color:#555;font-weight:600;padding:10px 14px;
   text-align:left;border-bottom:2px solid #e9ecef}
td{padding:8px 14px;border-bottom:1px solid #f4f4f4}
tr:hover td{background:#fafafa}
.lift-pos{color:#16a34a;font-weight:600}
.lift-neg{color:#dc2626;font-weight:600}
.sig{color:#16a34a;font-weight:700}
.ns{color:#9ca3af}
/* ── chart grid ── */
.chart-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(500px,1fr));gap:16px}
.chart-item{background:#fafafa;border-radius:10px;padding:14px;text-align:center;
            border:1px solid #f0f0f0}
.chart-item img{max-width:100%;border-radius:6px;cursor:zoom-in}
.chart-title{font-size:.8rem;color:#666;margin-top:8px}
.chart-insight{font-size:.8rem;color:#444;line-height:1.55;margin-top:8px;
               padding:7px 10px;background:#f0f4ff;border-radius:6px;text-align:left}
.ai-chip{display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);
         color:#fff;font-size:.7rem;padding:1px 7px;border-radius:10px;
         margin-right:5px;font-weight:600;vertical-align:middle}
/* ── badge ── */
.tag-badge{display:inline-block;padding:2px 9px;border-radius:12px;font-size:.78rem;
           font-weight:600;margin-right:3px}
.b-ar{background:#dbeafe;color:#1d4ed8}
.b-aw{background:#dcfce7;color:#15803d}
.b-fv{background:#fce7f3;color:#9d174d}
.b-in{background:#ffedd5;color:#c2410c}
/* ── lightbox overlay ── */
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);
    z-index:9999;cursor:pointer;place-items:center}
#lb.open{display:grid}
#lb img{max-width:95vw;max-height:92vh;border-radius:8px;box-shadow:0 0 40px rgba(0,0,0,.5)}
/* ── methodology steps ── */
.meth-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:18px}
.meth-step{background:#f8f9fa;border-radius:10px;padding:16px;border-top:4px solid #667eea;
           position:relative}
.meth-step.active-step{background:linear-gradient(135deg,#667eea08,#764ba208);
                        border-top-color:#27ae60}
.meth-step .step-num{font-size:.72rem;font-weight:700;color:#667eea;letter-spacing:.5px;
                      text-transform:uppercase;margin-bottom:4px}
.meth-step h4{font-size:.92rem;color:#1a1a2e;margin-bottom:6px}
.meth-step p{font-size:.82rem;color:#555;line-height:1.55;margin-bottom:8px}
.meth-config{display:inline-block;background:#1a1a2e;color:#a8dadc;font-size:.78rem;
             padding:3px 9px;border-radius:12px;font-family:monospace;font-weight:600}
.meth-config.on{background:#16a34a;color:#fff}
.meth-config.off{background:#94a3b8;color:#fff}
.meth-config.warn{background:#f59e0b;color:#fff}
.meth-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px}
.meth-principle{font-size:.78rem;color:#667eea;font-style:italic;margin-top:4px;border-top:1px solid #e2e8f0;padding-top:4px}
"""

_BADGE = {
    "AD_RECALL":    '<span class="tag-badge b-ar">AD_RECALL</span>',
    "AWARENESS":    '<span class="tag-badge b-aw">AWARENESS</span>',
    "FAVORABILITY": '<span class="tag-badge b-fv">FAVORABILITY</span>',
    "INTENT":       '<span class="tag-badge b-in">INTENT</span>',
}
_JS = """
const lb=document.getElementById('lb');
document.querySelectorAll('.chart-item img').forEach(img=>{
  img.addEventListener('click',()=>{
    lb.querySelector('img').src=img.src;lb.classList.add('open');});
});
lb.addEventListener('click',()=>lb.classList.remove('open'));
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _img_tag(path: str) -> str:
    return f'<img src="data:image/png;base64,{_b64(path)}" loading="lazy">'

def _lift_td(v) -> str:
    if pd.isna(v): return "<td>—</td>"
    cls = "lift-pos" if float(v) >= 0 else "lift-neg"
    return f'<td class="{cls}">{float(v):+.3f}</td>'

def _sig_td(s: str) -> str:
    return f'<td class="{"sig" if s == "✓" else "ns"}">{s}</td>'

def _badge(tag: str) -> str:
    return _BADGE.get(tag, f"<span>{tag}</span>")


# ═══════════════════════════════════════════════════════════════════════════════
#  Methodology card
# ═══════════════════════════════════════════════════════════════════════════════

def _methodology_card(config: dict, n_tests: int = 0, n_sig: int = 0) -> str:
    """
    Returns an HTML card explaining the 4-layer false-positive control pipeline,
    with current run settings highlighted.
    """
    alpha      = config.get("alpha",        0.10)
    min_n      = config.get("min_n",        5)
    weight     = config.get("weight_col",  "iv_weight")
    mt_corr    = config.get("mt_correction", "none")
    wlo        = config.get("winsorize_lower", 0.02)
    whi        = config.get("winsorize_upper", 0.98)

    # -- Weight description ---------------------------------------------------
    weight_desc = (
        "1/SE² (inverse-variance) &mdash; tighter estimates carry more weight"
        if weight == "iv_weight"
        else "harmonic-N &mdash; equal weight per campaign"
    )

    # -- MT correction description -------------------------------------------
    mt_label = {
        "none":       ("off",  "No correction applied &mdash; raw p-values used."),
        "bh":         ("on",   "Benjamini-Hochberg FDR &mdash; controls expected false discovery rate (q = α)."),
        "bonferroni": ("warn", "Bonferroni &mdash; family-wise error rate; conservative for many tests."),
        "holm":       ("on",   "Holm step-down &mdash; less conservative than Bonferroni, controls FWER."),
    }.get(mt_corr, ("off", mt_corr))
    mt_cls, mt_desc = mt_label

    # -- Sig rate (live from this run) ----------------------------------------
    sig_rate_html = ""
    if n_tests > 0:
        rate = 100 * n_sig / n_tests
        color = "#16a34a" if rate < 30 else ("#f59e0b" if rate < 60 else "#dc2626")
        sig_rate_html = (
            f'<div style="margin-top:10px;font-size:.82rem;color:#555">'  
            f'This run: <strong style="color:{color}">{n_sig}/{n_tests}</strong> '
            f'group-tests significant ({rate:.0f}%) &mdash; '
            + ("plausible rate" if rate < 30 else ("moderate; consider BH correction" if rate < 60 else "high rate; MT correction recommended"))
            + "</div>"
        )

    return f"""
<div class="card">
  <h2>🧪 Statistical Methodology &mdash; False Positive Control</h2>
  <p style="font-size:.88rem;color:#555;margin-bottom:16px">
    Four independent layers guard against spurious findings.
    Settings in <code>config.yaml</code> control each layer;
    current-run values are shown below.
  </p>

  <div class="meth-flow">

    <!-- Layer 1: Data Quality Guard -->
    <div class="meth-step active-step">
      <div class="step-num">Layer 1</div>
      <h4>🛡 Data Quality Guard</h4>
      <p>Removes extreme outliers and enforces minimum group sizes before any test is run.</p>
      <div class="meth-row">
        <span class="meth-config on">Winsorize P{int(wlo*100)}/P{int(whi*100)}</span>
        <span class="meth-config on">min n = {min_n}</span>
      </div>
      <div class="meth-principle">
        Clips lift at P{int(wlo*100)}&ndash;P{int(whi*100)} to reduce outlier leverage.
        Groups with &lt;{min_n} campaigns are dropped &mdash; too small to trust.
      </div>
    </div>

    <!-- Layer 2: Per-group significance test -->
    <div class="meth-step active-step">
      <div class="step-num">Layer 2</div>
      <h4>📊 Per-group Z-test</h4>
      <p>Each group is tested independently: does its weighted-mean lift differ from zero?</p>
      <div class="meth-row">
        <span class="meth-config on">α = {alpha}</span>
        <span class="meth-config on">weight = {weight}</span>
        <span class="meth-config on">two-sided</span>
      </div>
      <div class="meth-principle">
        H₀: μₐᵉᵗ = 0. Z-statistic = μ̂ / SÊ.
        Weight = {weight_desc}.
        CI = μ̂ &plusmn; zₐ/₂ &middot; SÊ.
      </div>
    </div>

    <!-- Layer 3: Cross-group test -->
    <div class="meth-step active-step">
      <div class="step-num">Layer 3</div>
      <h4>🔄 Cross-group Test</h4>
      <p>Tests whether groups within a cut differ from each other &mdash; not just from zero.</p>
      <div class="meth-row">
        <span class="meth-config on">Welch t (2 groups)</span>
        <span class="meth-config on">Tukey HSD (3+)</span>
      </div>
      <div class="meth-principle">
        2 groups: Welch&rsquo;s t-test (unequal variance assumed).
        3+ groups with sig. cross-p: Tukey HSD post-hoc identifies which pairs differ.
        Prevents cherry-picking the &ldquo;best&rdquo; group without acknowledging others.
      </div>
    </div>

    <!-- Layer 4: Multiple Testing Correction -->
    <div class="meth-step {'active-step' if mt_corr != 'none' else ''}">
      <div class="step-num">Layer 4</div>
      <h4>🔬 MT Correction</h4>
      <p>Adjusts p-values across all group&times;metric tests to control false discovery rate.</p>
      <div class="meth-row">
        <span class="meth-config {mt_cls}">{mt_corr.upper() if mt_corr != 'none' else 'OFF'}</span>
      </div>
      <div class="meth-principle">{mt_desc}</div>
    </div>

  </div>

  <!-- Quick-reference table -->
  <details style="margin-top:4px">
    <summary style="cursor:pointer;font-size:.85rem;color:#667eea;font-weight:600">
      ▶ MT Correction options reference
    </summary>
    <table style="margin-top:10px;font-size:.82rem">
      <thead><tr><th>Method</th><th>Controls</th><th>When to use</th><th>Trade-off</th></tr></thead>
      <tbody>
        <tr {'style="background:#f0fdf4"' if mt_corr=='none' else ''}>
          <td><strong>none</strong></td><td>nothing</td>
          <td>Exploratory; few tests; strong domain priors</td>
          <td>Inflated FP with many tests</td></tr>
        <tr {'style="background:#f0fdf4"' if mt_corr=='bh' else ''}>
          <td><strong>BH (FDR)</strong></td><td>False Discovery Rate</td>
          <td>Many tests; want to limit false discoveries</td>
          <td>Allows some FP; most powerful</td></tr>
        <tr {'style="background:#f0fdf4"' if mt_corr=='holm' else ''}>
          <td><strong>Holm</strong></td><td>Family-Wise Error Rate</td>
          <td>Moderate # tests; stricter than BH</td>
          <td>Less power than BH</td></tr>
        <tr {'style="background:#f0fdf4"' if mt_corr=='bonferroni' else ''}>
          <td><strong>Bonferroni</strong></td><td>Family-Wise Error Rate</td>
          <td>Few tests; must guarantee no FP</td>
          <td>Very conservative; low power</td></tr>
      </tbody>
    </table>
  </details>

  {sig_rate_html}
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  AI helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_prompt(results: dict, fi_result, config: dict) -> str:
    """Overall executive summary prompt — quantitative, data-driven."""
    alpha   = config.get("alpha", 0.10)
    mt_corr = config.get("mt_correction", "none")
    weight  = config.get("weight_col", "iv_weight")

    # Aggregate stats
    all_rows, sig_rows = [], []
    for cut_name, df in results.items():
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            if row.get("question_tag", "All Questions") == "All Questions":
                continue
            all_rows.append(row)
            if str(row.get("sig_vs_zero", "")) in ("✓", "?"):
                sig_rows.append({**row.to_dict(), "cut": cut_name})

    n_total = len(all_rows)
    n_sig   = len(sig_rows)
    sig_pct = 100 * n_sig / n_total if n_total > 0 else 0

    # Top significant results by lift magnitude
    sig_rows_sorted = sorted(sig_rows, key=lambda r: abs(float(r.get("weighted_mean_lift", 0))), reverse=True)

    data_lines = [
        f"Dataset: {n_total} group-metric tests across {len(results)} cuts; "
        f"α={alpha}, weight={weight}, MT={mt_corr}.",
        f"Significant results: {n_sig}/{n_total} ({sig_pct:.1f}%).",
        "",
        "Top significant results (cut / group / metric : lift pp, n, p-value):",
    ]
    for r in sig_rows_sorted[:20]:
        lift = float(r.get("weighted_mean_lift", 0))
        p    = r.get("p_vs_zero", float("nan"))
        ci_cols = [c for c in r if c.startswith("CI") and "lower" in c]
        ci_lo = float(r[ci_cols[0]]) if ci_cols else float("nan")
        ci_hi_cols = [c for c in r if c.startswith("CI") and "upper" in c]
        ci_hi = float(r[ci_hi_cols[0]]) if ci_hi_cols else float("nan")
        ci_str = f", 90%CI=[{ci_lo:+.3f},{ci_hi:+.3f}]" if not (ci_lo != ci_lo) else ""
        cross = r.get("cross_group_sig", "")
        cross_str = f", cross-group sig={cross}" if cross else ""
        data_lines.append(
            f"  {r['cut']} / {r.get('group','?')} / {r.get('question_tag','?')}: "
            f"lift={lift:+.4f}pp, n={int(r.get('n',0))}, p={float(p):.4f}{ci_str}{cross_str}"
        )

    # Non-significant cuts summary
    non_sig_cuts = [c for c, df in results.items()
                    if df is not None and len(df) > 0 and
                    not any(str(row.get("sig_vs_zero","")) in ("✓","?")
                            for _, row in df.iterrows()
                            if row.get("question_tag","") != "All Questions")]
    if non_sig_cuts:
        data_lines.append(f"No significant effects: {', '.join(non_sig_cuts)}.")

    # FI
    if fi_result is not None:
        items = fi_result.items() if isinstance(fi_result, dict) else [("Combined", fi_result)]
        for tag_name, imp_df in items:
            if imp_df is not None and len(imp_df) > 0:
                top5 = [(row["feature"], float(row["importance"])) for _, row in imp_df.head(5).iterrows()]
                top5_str = ", ".join(f"{f}({i:.3f})" for f, i in top5)
                data_lines.append(f"RF feature importance ({tag_name}): {top5_str}.")

    instructions = [
        "",
        "Write a 3-paragraph executive summary for a data-savvy marketing analytics audience:",
        "§1: Lead with the headline stat (sig rate %). Call out the strongest 2-3 levers "
        "with exact lift values (e.g. '+0.052pp') and sample sizes. "
        "Note if cross-group differences are significant.",
        "§2: Contrast top-performing vs bottom-performing segments with specific numbers. "
        "Mention confidence intervals where informative. "
        "Reference feature importance ranking if available.",
        "§3: Give ONE concrete, quantified recommendation a campaign manager can act on "
        "(e.g. 'Shift budget toward 2.5-5s watch-time creatives, which drive +X.XXpp lift '",
        "across N campaigns').",
        "Rules: no fluffy intro phrases ('In summary…', 'Overall…'). "
        "Start §1 with the sig-rate stat. Use exact numbers. ≤4 sentences per paragraph.",
    ]

    full = "\n".join(data_lines + instructions)
    if len(full) > 14_000:
        header = "\n".join(data_lines[:6])
        footer = "\n".join(instructions)
        max_data = 14_000 - len(header) - len(footer) - 60
        data_mid = "\n".join(data_lines[6:])
        full = header + "\n" + data_mid[:max_data] + "\n[...truncated...]\n" + footer
    return full


def _build_per_cut_prompt(results: dict, config: dict) -> str:
    """
    Single prompt asking the AI to produce one 1-2 sentence insight per cut.
    Returns a prompt expecting JSON: {"CutName": "insight sentence", ...}
    """
    alpha = config.get("alpha", 0.10)
    lines = [
        "You are a senior data scientist analyzing TikTok Brand Lift Study results.",
        f"Statistical threshold: α={alpha}. Lift = absolute percentage-point change in survey metric.",
        "Below are results per analysis cut. For each cut, provide ONE insight sentence (≤25 words).",
        "Focus on: which group wins, the lift magnitude, whether the cross-group difference is significant.",
        "If no groups are significant, note that explicitly.",
        "Return ONLY a JSON object mapping cut name → insight string. No extra text.",
        "",
        "Results:",
    ]
    for cut_name, df in results.items():
        if df is None or len(df) == 0:
            continue
        lines.append(f"\n[{cut_name}]")
        all_g = df[df.get("question_tag", pd.Series(dtype=str)).ne("All Questions")] if "question_tag" in df.columns else df
        for _, row in all_g.iterrows():
            sig   = "✓" if str(row.get("sig_vs_zero","")) in ("✓","?") else "ns"
            lift  = float(row.get("weighted_mean_lift", 0))
            cross = row.get("cross_group_sig", "")
            lines.append(
                f"  {row.get('group','?')} | {row.get('question_tag','?')}: "
                f"lift={lift:+.3f}pp, sig={sig}"
                + (f", cross_sig={cross}" if cross else "")
            )
    return "\n".join(lines)


def _build_heatmap_prompt(results: dict, config: dict) -> str:
    """
    Build a prompt asking the AI to interpret the summary heatmap data:
    what it shows, key patterns, top performers, and actionable insights.
    """
    alpha       = config.get("alpha", 0.10)
    sel_tags    = config.get("question_tags", [])

    rows_by_cut: dict = {}
    for cut_name, res_df in results.items():
        if res_df is None or len(res_df) == 0:
            continue
        cut_rows = []
        for tag in sel_tags:
            if "question_tag" in res_df.columns:
                df_f = res_df[res_df["question_tag"] == tag]
            else:
                df_f = res_df
            for _, row in df_f.iterrows():
                lift = float(row.get("weighted_mean_lift", float("nan")))
                sig  = "*" if str(row.get("sig_vs_zero", "")) == "✓" else ""
                cut_rows.append(
                    f"  [{tag}] {row.get('group','?')}: {lift:+.3f}pp{sig}"
                )
        if cut_rows:
            rows_by_cut[cut_name] = cut_rows

    if not rows_by_cut:
        return ""

    data_lines = []
    for cut_name, rows in rows_by_cut.items():
        data_lines.append(f"\n### {cut_name}")
        data_lines.extend(rows)

    prompt = (
        "You are a senior marketing data scientist interpreting a TikTok Brand Lift Study "
        "meta-analysis heatmap.\n"
        f"Statistical threshold: \u03b1={alpha}. Lift values are weighted mean absolute "
        "percentage-point change in survey metrics (* = statistically significant vs 0).\n"
        "The heatmap rows are question tags, columns are cut groups (e.g. Watch Time buckets, "
        "VCR buckets, etc.), values are weighted mean lift.\n\n"
        "Data:\n" + "\n".join(data_lines) + "\n\n"
        "Please provide a concise HTML analysis (\u2264180 words total) covering:\n"
        "1. What this heatmap shows (1-2 sentences explaining rows/columns/values).\n"
        "2. Top-performing cut groups (highest lift, especially if significant).\n"
        "3. Cross-tag patterns (e.g. a group that wins across all metrics).\n"
        "4. Practical implication for campaign optimization (2-3 bullet points).\n"
        "Format: HTML paragraphs and a \u003cul\u003e list for bullet points. No markdown."
    )
    return prompt


def _read_http_error(e) -> str:
    """Extract meaningful message from urllib HTTPError response body."""
    import json as _j
    prefix = f"HTTP {e.code}"
    try:
        body = e.fp.read().decode("utf-8", errors="replace") if hasattr(e, 'fp') and e.fp else e.read().decode("utf-8", errors="replace")
        parsed = _j.loads(body)
        # Anthropic format: {"type":"error","error":{"type":"...","message":"..."}}
        if isinstance(parsed.get("error"), dict) and "message" in parsed["error"]:
            return f"{prefix}: {parsed['error']['message']}"
        # OpenAI / Groq format: {"error":{"message":"..."}}
        if "error" in parsed:
            err = parsed["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            return f"{prefix}: {msg[:300]}"
        return f"{prefix}: {body[:300]}"
    except Exception as parse_exc:
        return f"{prefix} {e.reason} (body parse failed: {parse_exc})"


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    import urllib.request, json as _j, urllib.error as _ue
    payload = _j.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 600,
        "temperature": 0.5,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _j.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except _ue.HTTPError as e:
        raise RuntimeError(_read_http_error(e)) from e


def _call_anthropic(prompt: str, api_key: str, model: str) -> str:
    import urllib.request, json as _j, urllib.error as _ue
    payload = _j.dumps({
        "model": model,
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"x-api-key": api_key,
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _j.loads(r.read())
        return data["content"][0]["text"].strip()
    except _ue.HTTPError as e:
        raise RuntimeError(_read_http_error(e)) from e


def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Google Gemini API — free tier available at aistudio.google.com."""
    import urllib.request, json as _j, urllib.error as _ue
    # Strip provider prefix if user typed e.g. "google/gemini-2.0-flash"
    model_id = model.split("/")[-1]
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model_id}:generateContent?key={api_key}")
    payload = _j.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 700, "temperature": 0.5},
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _j.loads(r.read())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except _ue.HTTPError as e:
        raise RuntimeError(_read_http_error(e)) from e


def _call_groq(prompt: str, api_key: str, model: str) -> str:
    """Groq free-tier API (Llama / Gemma models)."""
    import urllib.request, json as _j, urllib.error as _ue
    # Groq uses OpenAI-compatible endpoint
    model_id = model.split("/")[-1]
    payload = _j.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 700,
        "temperature": 0.5,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _j.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except _ue.HTTPError as e:
        raise RuntimeError(_read_http_error(e)) from e


def _detect_provider(api_key: str, model: str) -> str:
    """
    Detect provider — API key format takes priority over model name.
    Supported prefixes:
      AIza...        → Gemini (Google AI Studio)
      sk-ant...      → Anthropic (Claude)
      gsk_...        → Groq (Llama / Gemma free tier)
      sk-proj-...    → OpenAI (project key, newer format)
      sk-o1-...      → OpenAI (o1 key)
      sk-...         → OpenAI (classic key)
    Unknown prefix   → assume OpenAI-compatible endpoint
    """
    # Key-based detection first (unambiguous)
    if api_key.startswith("AIza"):
        return "gemini"
    if api_key.startswith("sk-ant"):
        return "anthropic"
    if api_key.startswith("gsk_"):
        return "groq"
    if api_key.startswith(("sk-proj-", "sk-o1-", "sk-")):
        return "openai"
    # Fall back to model name when key format is unknown
    m = model.lower()
    if "gemini" in m:
        return "gemini"
    if "claude" in m:
        return "anthropic"
    if "llama" in m or "mixtral" in m or "gemma" in m or "qwen" in m:
        return "groq"
    return "openai"  # default: try OpenAI-compatible endpoint


_PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",             # Haiku 4.5 - cheapest current claude
    "gemini":    "gemini-2.0-flash",             # free tier
    "groq":      "llama-3.3-70b-versatile",      # free tier, better quality
    "openai":    "gpt-4o-mini",                  # cheapest openai
}


def _validate_model(provider: str, model: str) -> str:
    """Return model if it matches provider, else return provider's default."""
    m = model.lower()
    if provider == "anthropic" and not any(x in m for x in ("claude", "haiku", "sonnet", "opus", "fable", "mythos")):
        return _PROVIDER_DEFAULT_MODELS["anthropic"]
    if provider == "gemini" and "gemini" not in m:
        return _PROVIDER_DEFAULT_MODELS["gemini"]
    if provider == "groq" and not any(x in m for x in ("llama", "mixtral", "gemma", "qwen", "deepseek")):
        return _PROVIDER_DEFAULT_MODELS["groq"]
    if provider == "openai" and not any(x in m for x in ("gpt", "o1", "o3", "chatgpt", "davinci", "babbage")):
        return _PROVIDER_DEFAULT_MODELS["openai"]
    return model


# Sentinel objects so the caller knows WHY the AI call failed
class _AIQuotaError(Exception): pass
class _AIError(Exception): pass


def _call_ai_raw(prompt: str, api_key: str, model: str) -> str:
    """Like _call_ai but returns plain text (not HTML-wrapped paragraphs)."""
    import time
    provider = _detect_provider(api_key, model)
    model    = _validate_model(provider, model)
    _fn = {"gemini": _call_gemini, "anthropic": _call_anthropic,
           "groq": _call_groq}.get(provider, _call_openai)
    last_exc: Exception = Exception("unknown")
    for attempt in range(2):
        try:
            return _fn(prompt, api_key, model)
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                if attempt == 0:
                    time.sleep(3)
                    continue
                raise _AIQuotaError(str(exc)) from exc
            raise _AIError(str(exc)) from exc
    raise _AIError(str(last_exc))


def _call_ai(prompt: str, api_key: str, model: str):
    """
    Call AI API.
    Returns  : formatted HTML str   on success
    Raises   : _AIQuotaError        on 429 / rate-limit / quota exhausted
               _AIError             on any other failure
    """
    import time
    provider = _detect_provider(api_key, model)
    model   = _validate_model(provider, model)   # ensure model matches provider
    _fn = {"gemini": _call_gemini, "anthropic": _call_anthropic,
           "groq": _call_groq}.get(provider, _call_openai)

    last_exc: Exception = Exception("unknown")
    for attempt in range(2):          # 1 retry on rate-limit
        try:
            text = _fn(prompt, api_key, model)
            return "\n".join(
                f"<p>{p.strip()}</p>"
                for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()
            )
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            detail = f"[{provider}/{model}] {exc}"
            is_quota = ("429" in err_str or "quota" in err_str
                        or "rate" in err_str or "resource_exhausted" in err_str)
            if is_quota and attempt == 0:
                time.sleep(3)
                continue
            if is_quota:
                raise _AIQuotaError(detail) from exc
            raise _AIError(detail) from exc

    raise _AIQuotaError(f"[{provider}/{model}] {last_exc}")


def _template_summary(results: dict, fi_result, n_sig: int, n_total: int) -> str:
    top_rows = []
    for cut_name, df in results.items():
        if df is None or len(df) == 0:
            continue
        for _, row in df.iterrows():
            if row.get("question_tag") == "All Questions" or row.get("sig_vs_zero") != "✓":
                continue
            top_rows.append({"cut": cut_name, "group": row.get("group",""),
                              "tag": row.get("question_tag",""),
                              "lift": float(row.get("weighted_mean_lift", 0))})
    if not top_rows:
        return "<p>No significant effects detected at the configured significance level.</p>"
    top = sorted(top_rows, key=lambda x: abs(x["lift"]), reverse=True)[:3]
    pct = f"{100 * n_sig / n_total:.0f}%" if n_total > 0 else "N/A"
    p1 = (f"<p>The meta-analysis found <strong>{n_sig} significant group-level effects</strong> "
          f"({pct} of {n_total} tests), indicating measurable variation in lift across "
          f"campaign attributes.</p>")
    top1 = top[0]
    p2_parts = [f"<strong>{top1['cut']} / {top1['group']}</strong> "
                f"({top1['tag']}: {top1['lift']:+.3f}pp)"]
    if len(top) > 1:
        p2_parts.append(f"<strong>{top[1]['cut']} / {top[1]['group']}</strong> "
                        f"({top[1]['tag']}: {top[1]['lift']:+.3f}pp)")
    p2 = f"<p>Top effects: {' and '.join(p2_parts)}.</p>"
    fi_note = ""
    if fi_result is not None:
        imp = next(iter(fi_result.values()), None) if isinstance(fi_result, dict) else fi_result
        if imp is not None and len(imp) > 0:
            fi_note = (f" Feature importance analysis identifies "
                       f"<strong>{imp.iloc[0]['feature']}</strong> as the top predictor of lift.")
    p3 = (f"<p>To improve Brand Lift, focus on the highest-performing segments identified above.{fi_note} "
          f"Full cut-by-cut details are in the sections below.</p>")
    return p1 + p2 + p3


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(
    output_dir: str,
    results: dict,
    config: dict,
    fi_result=None,
    n_campaigns: int = 0,
    ai_api_key: Optional[str] = None,
    ai_model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a self-contained HTML report.

    Parameters
    ----------
    output_dir   : directory where PNGs live and where report will be saved
    results      : {cut_name: DataFrame}  from pipeline._results
    config       : analysis config dict
    fi_result    : DataFrame or {tag: DataFrame} or None
    n_campaigns  : campaign count for KPI display
    ai_api_key   : OpenAI or Anthropic key (optional)
    ai_model     : model name (default gpt-4o-mini)

    Returns
    -------
    str  absolute path to generated HTML
    """
    ts       = datetime.datetime.now()
    ts_str   = ts.strftime("%Y-%m-%d %H:%M")
    out_fname = ts.strftime("report_%Y%m%d_%H%M%S.html")
    out_path  = os.path.join(output_dir, out_fname)

    # ── Collect PNGs ──────────────────────────────────────────────────────────
    all_pngs = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir) if f.lower().endswith(".png")
    ]) if os.path.isdir(output_dir) else []

    cut_pngs   = [p for p in all_pngs if not any(
        os.path.basename(p).startswith(x) for x in
        ("Pipeline_", "Heatmap_", "cross_"))]
    fi_pngs    = [p for p in all_pngs if "Feature_Importance" in p or "Pipeline_FI" in p]
    heat_pngs  = [p for p in all_pngs if os.path.basename(p).startswith("Heatmap_")]
    cross_pngs = [p for p in all_pngs if os.path.basename(p).startswith("cross_")]
    # Remove FI from cut_pngs
    cut_pngs   = [p for p in cut_pngs if p not in fi_pngs]

    # ── Aggregate results ─────────────────────────────────────────────────────
    all_rows = []
    for cut_name, df in results.items():
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                all_rows.append({**row.to_dict(), "cut": cut_name})
    adf     = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    n_total = len(adf[adf.get("question_tag", pd.Series(dtype=str)) != "All Questions"]) if len(adf) > 0 else 0
    n_sig   = int((adf[adf.get("question_tag", pd.Series(dtype=str)) != "All Questions"]
                   .get("sig_vs_zero", pd.Series(dtype=str)) == "✓").sum()) if n_total > 0 else 0
    n_cuts  = sum(1 for _, df in results.items() if df is not None and len(df) > 0)

    # ── Resolve effective API key (user > built-in > template) ────────
    key_raw = (ai_api_key or "").strip()
    if not key_raw and _BUILTIN_AI_KEY:
        key_raw  = _BUILTIN_AI_KEY
        ai_model = _BUILTIN_AI_MODEL

    # ── Summary text ───────────────────────────────────────────────
    ai_notice  = ""   # optional warning banner above summary
    ai_result  = None

    # dict: cut_name -> "insight sentence" (populated by per-cut AI call)
    per_cut_insights: dict = {}

    if key_raw:
        prompt = _build_prompt(results, fi_result, config)
        try:
            ai_result = _call_ai(prompt, key_raw, ai_model)
        except _AIQuotaError:
            ai_notice = (
                '<div style="background:#fff8e1;border:1px solid #f9a825;'
                'border-radius:8px;padding:12px 16px;margin-bottom:14px;'
                'font-size:.9rem;color:#5d4037">'
                '<strong>&#9888; AI quota temporarily exhausted</strong> &mdash; '
                'The AI model has reached its rate limit. '
                'Showing auto-generated template summary instead. '
                'Please try again later, or add your own API key in Settings.'
                '</div>'
            )
        except _AIError as exc:
            ai_notice = (
                '<div style="background:#fce4ec;border:1px solid #e91e63;'
                'border-radius:8px;padding:12px 16px;margin-bottom:14px;'
                'font-size:.9rem;color:#880e4f">'
                + f'<strong>&#10060; AI call failed</strong> &mdash; {exc}<br>'
                + 'Showing auto-generated template summary instead. '
                + 'Please check your API key or try again later.'
                + '</div>'
            )

        # ── Per-cut insights (second AI call) ─────────────────────────────────
        if results:
            try:
                import json as _json
                pc_prompt = _build_per_cut_prompt(results, config)
                pc_raw    = _call_ai_raw(pc_prompt, key_raw, ai_model)
                # Strip markdown fences if any
                pc_clean  = pc_raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                pc_parsed = _json.loads(pc_clean)
                if isinstance(pc_parsed, dict):
                    per_cut_insights = {str(k): str(v) for k, v in pc_parsed.items()}
            except Exception:
                pass  # per-cut summaries are best-effort; silently skip on failure

        # -- Heatmap interpretation (third AI call) -------------------------------------------
        heatmap_ai_html = ""
        try:
            hm_prompt = _build_heatmap_prompt(results, config)
            if hm_prompt:
                heatmap_ai_html = _call_ai(hm_prompt, key_raw, ai_model)
        except Exception:
            pass  # heatmap interpretation is best-effort

    if ai_result is not None:
        is_builtin = bool(_BUILTIN_AI_KEY and key_raw == _BUILTIN_AI_KEY)
        suffix   = " (Gemini Flash)" if is_builtin else ""
        ai_label = "✨ AI Executive Summary" + suffix
        sum_html = ai_result
    else:
        sum_html = _template_summary(results, fi_result, n_sig, n_total)
        ai_label = "📋 Summary"
    # ── Meta ──────────────────────────────────────────────────────────────────
    alpha    = config.get("alpha", 0.10)
    weight   = config.get("weight_col", "iv_weight")
    mt_corr  = config.get("mt_correction", "none")
    tags_str = ", ".join(config.get("question_tags", []))
    _meth_cfg = {
        "alpha":           alpha,
        "min_n":           config.get("min_n", 5),
        "weight_col":      weight,
        "mt_correction":   mt_corr,
        "winsorize_lower": config.get("winsorize_lower", 0.02),
        "winsorize_upper": config.get("winsorize_upper", 0.98),
    }

    # ═════════════════════════════════════════════════════════════════════════
    #  HTML construction
    # ═════════════════════════════════════════════════════════════════════════
    H = []

    H.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>BLS Meta-Analysis Report — {ts_str}</title>
<style>{_CSS}</style>
</head>
<body>
<!-- lightbox -->
<div id="lb"><img src=""></div>

<div class="container">

<!-- HEADER -->
<div class="header">
  <h1>📊 BLS Meta-Analysis Report</h1>
  <div class="meta">
    <span>🕐 {ts_str}</span>
    <span>α = {alpha}</span>
    <span>weight = {weight}</span>
    <span>MT = {mt_corr}</span>
    <span>tags = {tags_str}</span>
  </div>
</div>

<!-- KPIs -->
<div class="card">
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-value">{n_campaigns:,}</div>
      <div class="kpi-label">Campaigns</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">{n_cuts}</div>
      <div class="kpi-label">Cuts Analyzed</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">{n_total}</div>
      <div class="kpi-label">Group Tests</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-value">{n_sig}</div>
      <div class="kpi-label">Significant ✓</div>
    </div>
  </div>
</div>

<!-- SUMMARY -->
<div class="card">
  <h2>{ai_label}</h2>
  {ai_notice}<div class="ai-box">{sum_html}</div>
</div>
""")

    # ── Methodology card ──────────────────────────────────────────────────────
    H.append(_methodology_card(_meth_cfg, n_total, n_sig))

    # ── Key findings table ────────────────────────────────────────────────────
    if not adf.empty and "sig_vs_zero" in adf.columns:
        top_sig = (adf[(adf["sig_vs_zero"] == "✓") &
                       (adf.get("question_tag", pd.Series(dtype=str)) != "All Questions")]
                   .sort_values("weighted_mean_lift", ascending=False)
                   .head(12))
        if len(top_sig) > 0:
            rows_html = ""
            for _, row in top_sig.iterrows():
                tag   = row.get("question_tag", "")
                lift  = row.get("weighted_mean_lift", float("nan"))
                p_val = row.get("p_vs_zero", float("nan"))
                p_cor = row.get("p_corrected", float("nan"))
                p_disp = (f"{float(p_cor):.4f}" if not pd.isna(p_cor)
                          else (f"{float(p_val):.4f}" if not pd.isna(p_val) else "—"))
                rows_html += (
                    f"<tr><td>{row.get('cut','')}</td>"
                    f"<td><strong>{row.get('group','')}</strong></td>"
                    f"<td>{_badge(tag)}</td>"
                    f"{_lift_td(lift)}"
                    f"<td>{int(row.get('n',0))}</td>"
                    f"<td>{p_disp}</td>"
                    f"{_sig_td(row.get('sig_vs_zero','ns'))}</tr>"
                )
            H.append(f"""
<div class="card">
  <h2>🎯 Top Significant Groups</h2>
  <table>
    <thead>
      <tr><th>Cut</th><th>Group</th><th>Metric</th>
          <th>Lift (pp)</th><th>n</th><th>p-value</th><th>Sig</th></tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""")

    # ── Cut charts ────────────────────────────────────────────────────────────
    if cut_pngs:
        def _chart_insight_html(png_path):
            fname = os.path.basename(png_path)[:-4]
            best_key, best_len = None, 0
            for cut_key in per_cut_insights:
                norm = cut_key.replace(" ", "_").replace("-", "_")
                if fname.startswith(norm) or norm in fname:
                    if len(norm) > best_len:
                        best_key, best_len = cut_key, len(norm)
            if not best_key:
                return ""
            return (
                f'<div class="chart-insight">'
                f'<span class="ai-chip">&#10024; AI</span> '
                f'{per_cut_insights[best_key]}</div>'
            )

        items = "".join(
            f'<div class="chart-item">{_img_tag(p)}'
            f'<div class="chart-title">{os.path.basename(p)[:-4].replace("_"," ")}</div>'
            f'{_chart_insight_html(p)}</div>'
            for p in cut_pngs
        )
        H.append(f"""
<div class="card">
  <h2>📈 Cut Analysis Charts</h2>
  <p style="font-size:.85rem;color:#888;margin-bottom:12px">Click any chart to enlarge.</p>
  <div class="chart-grid">{items}</div>
</div>
""")

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    if heat_pngs:
        items = "".join(
            f'<div class="chart-item">{_img_tag(p)}'
            f'<div class="chart-title">{os.path.basename(p)[:-4].replace("_"," ")}</div></div>'
            for p in heat_pngs
        )
        _hm_ai_block = ""
        if heatmap_ai_html:
            _hm_ai_block = (
                '<div class="ai-box" style="margin-bottom:16px">'
                '<div style="font-size:.8rem;color:#667eea;font-weight:600;margin-bottom:6px">'
                '&#10024; AI Heatmap Interpretation</div>'
                + heatmap_ai_html
                + '</div>'
            )
        H.append(f"""
<div class="card">
  <h2>🗺️ Summary Heatmaps</h2>
  <p style="font-size:.85rem;color:#888;margin-bottom:12px">
    Columns sorted by mean lift (high &rarr; low).
    Values = weighted mean lift (pp); <strong>*</strong> = sig vs 0 at &alpha;={alpha}.
    &#10003; in column label = sig difference across groups within that cut.
  </p>
  {_hm_ai_block}
  <div class="chart-grid">{items}</div>
</div>
""")

    # ── Feature importance ────────────────────────────────────────────────────
    if fi_pngs or fi_result is not None:
        fi_chart_html = "".join(
            f'<div class="chart-item">{_img_tag(p)}'
            f'<div class="chart-title">{os.path.basename(p)[:-4].replace("_"," ")}</div></div>'
            for p in fi_pngs
        )
        fi_table_html = ""
        if fi_result is not None:
            items_fi = (fi_result.items() if isinstance(fi_result, dict)
                        else [("Combined", fi_result)])
            for tag_name, imp_df in items_fi:
                if imp_df is None or len(imp_df) == 0:
                    continue
                rows = "".join(
                    f"<tr><td>{i+1}</td><td>{row['feature']}</td>"
                    f"<td>{float(row['importance']):.4f}</td></tr>"
                    for i, (_, row) in enumerate(imp_df.head(15).iterrows())
                )
                fi_table_html += (
                    f"<h3>Top Features — {tag_name}</h3>"
                    f"<table><thead><tr><th>#</th><th>Feature</th><th>Importance</th></tr></thead>"
                    f"<tbody>{rows}</tbody></table>"
                )
        H.append(f"""
<div class="card">
  <h2>🌲 Feature Importance</h2>
  {'<div class="chart-grid">' + fi_chart_html + '</div>' if fi_chart_html else ''}
  {fi_table_html}
</div>
""")

    # ── Cross-cuts ────────────────────────────────────────────────────────────
    if cross_pngs:
        items = "".join(
            f'<div class="chart-item">{_img_tag(p)}'
            f'<div class="chart-title">{os.path.basename(p)[:-4].replace("_"," ")}</div></div>'
            for p in cross_pngs
        )
        H.append(f"""
<div class="card">
  <h2>🔀 Cross-Cut Heatmaps</h2>
  <div class="chart-grid">{items}</div>
</div>
""")

    # ── Footer ────────────────────────────────────────────────────────────────
    H.append(f"""
<p style="text-align:center;color:#bbb;font-size:.8rem;padding:24px 0">
  BLS Meta-Analysis Pipeline &nbsp;·&nbsp; {ts_str}
</p>
</div>

<script>{_JS}</script>
</body>
</html>""")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(H))

    print(f"  [Report] Saved → {out_path}")
    return out_path
